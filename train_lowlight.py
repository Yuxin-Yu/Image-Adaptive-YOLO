#! /usr/bin/env python
# coding=utf-8


import os
import time
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from tqdm import tqdm
from core.dataset_lowlight import Dataset
from core.yolov3_lowlight import YOLOV3
from core.config_lowlight import cfg
from core.config_lowlight import args
import random
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

if args.use_gpu == 0:
    gpu_id = '-1'
else:
    gpu_id = args.gpu_id
    gpu_list = list()
    gpu_ids = gpu_id.split(',')
    for i in range(len(gpu_ids)):
        gpu_list.append('/gpu:%d' % int(i))
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id


exp_folder = os.path.join(args.exp_dir, 'exp_{}'.format(args.exp_num))

set_ckpt_dir = args.ckpt_dir
args.ckpt_dir = os.path.join(exp_folder, set_ckpt_dir)
if not os.path.exists(args.ckpt_dir):
    os.makedirs(args.ckpt_dir)

config_log = os.path.join(exp_folder, 'config.txt')
arg_dict = args.__dict__
msg = ['{}: {}\n'.format(k, v) for k, v in arg_dict.items()]
utils.write_mes(msg, config_log, mode='w')


class YoloTrain(object):
    def __init__(self):
        self.anchor_per_scale    = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes             = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes         = len(self.classes)
        self.learn_rate_init     = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end      = cfg.TRAIN.LEARN_RATE_END
        self.first_stage_epochs  = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_periods      = cfg.TRAIN.WARMUP_EPOCHS
        self.initial_weight      = cfg.TRAIN.INITIAL_WEIGHT
        self.time                = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay    = cfg.YOLO.MOVING_AVE_DECAY
        self.max_bbox_per_scale  = 150
        self.train_logdir        = "./data/log/train"
        self.trainset            = Dataset('train')
        self.testset             = Dataset('test')
        self.steps_per_period    = len(self.trainset)
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        self.end_points = {}
        self.brightness_aver = np.zeros(cfg.TRAIN.BATCH_SIZE)
        
        # self.sess                = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        with tf.compat.v1.name_scope('define_input'):
            if args.grayimage_FLAG:
                self.input_data   = tf.compat.v1.placeholder(tf.float32, [None, None, None, 1], name='input_data')
            else:
                self.input_data   = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='input_data')
            self.label_sbbox  = tf.compat.v1.placeholder(dtype=tf.float32, name='label_sbbox')
            self.label_mbbox  = tf.compat.v1.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox  = tf.compat.v1.placeholder(dtype=tf.float32, name='label_lbbox')
            self.true_sbboxes = tf.compat.v1.placeholder(dtype=tf.float32, name='sbboxes')
            self.true_mbboxes = tf.compat.v1.placeholder(dtype=tf.float32, name='mbboxes')
            self.true_lbboxes = tf.compat.v1.placeholder(dtype=tf.float32, name='lbboxes')
            self.input_data_clean   = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='input_data')

            self.trainable     = tf.compat.v1.placeholder(dtype=tf.bool, name='training')
            

        with tf.compat.v1.name_scope("define_loss"):
            self.model = YOLOV3(self.input_data, self.trainable, self.input_data_clean,self.brightness_aver)
            #t_variables = tf.compat.v1.trainable_variables()
            #print("t_variables", t_variables)
            self.end_points = self.model.get_endpoints()
            for i in self.end_points.keys():
                print(i, self.end_points[i].shape)
            # self.net_var = [v for v in t_variables if not 'extract_parameters' in v.name]
            self.net_var = tf.compat.v1.global_variables()
            self.giou_loss, self.conf_loss, self.prob_loss, self.recovery_loss = self.model.compute_loss(
                                                    self.label_sbbox,  self.label_mbbox,  self.label_lbbox,
                                                    self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
            # self.loss only includes the detection loss.
            self.loss = self.giou_loss + self.conf_loss + self.prob_loss
            self.test_loss = tf.Variable(0, dtype=tf.float64, trainable=False, name='test_loss')

        with tf.compat.v1.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                        dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant( (self.first_stage_epochs + self.second_stage_epochs)* self.steps_per_period,
                                        dtype=tf.float64, name='train_steps')
            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
                                    (1 + tf.cos(
                                        (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            )
            global_step_update = tf.compat.v1.assign_add(self.global_step, 1.0)

        with tf.compat.v1.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.compat.v1.trainable_variables())

        with tf.compat.v1.name_scope("define_first_stage_train"):
            self.first_stage_trainable_var_list = []
            for var in tf.compat.v1.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if var_name_mess[0] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
                    self.first_stage_trainable_var_list.append(var)

            first_stage_optimizer = tf.compat.v1.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=self.first_stage_trainable_var_list)
            with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()

        with tf.compat.v1.name_scope("define_second_stage_train"):
            second_stage_trainable_var_list = tf.compat.v1.trainable_variables()
            second_stage_optimizer = tf.compat.v1.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=second_stage_trainable_var_list)

            with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()

        with tf.compat.v1.name_scope('loader_and_saver'):
            self.loader = tf.compat.v1.train.Saver(self.net_var)
            self.saver  = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=5)

        with tf.compat.v1.name_scope('summary'):
            tf.compat.v1.summary.scalar("learn_rate",      self.learn_rate)
            tf.compat.v1.summary.scalar("giou_loss",  self.giou_loss)
            tf.compat.v1.summary.scalar("conf_loss",  self.conf_loss)
            tf.compat.v1.summary.scalar("prob_loss",  self.prob_loss)
            tf.compat.v1.summary.scalar("recovery_loss",  self.recovery_loss)
            tf.compat.v1.summary.scalar("total_loss", self.loss)
            tf.compat.v1.summary.scalar("test_loss", self.test_loss)

            # logdir = "./data/log/"
            logdir = os.path.join(exp_folder, 'log')

            if os.path.exists(logdir): shutil.rmtree(logdir)
            os.mkdir(logdir)
            self.write_op = tf.compat.v1.summary.merge_all()
            self.summary_writer  = tf.compat.v1.summary.FileWriter(logdir, graph=self.sess.graph)


    def train(self):
        self.sess.run(tf.compat.v1.global_variables_initializer())
        try:
            print('=> Restoring weights from: %s ... ' % self.initial_weight)
            self.loader.restore(self.sess, self.initial_weight)
        except:
            print('=> %s does not exist !!!' % self.initial_weight)
            print('=> Now it starts to train YOLOV3 from scratch ...')
            self.first_stage_epochs = 0

        train_epoch_loss_plot, test_epoch_loss_plot = [0], [0]

        for epoch in range(1, 1+self.first_stage_epochs+self.second_stage_epochs):
            if epoch <= self.first_stage_epochs:
                train_op = self.train_op_with_frozen_variables
            else:
                train_op = self.train_op_with_all_variables

            pbar = tqdm(self.trainset)
            train_epoch_loss, test_epoch_loss = [], []

            for train_data in pbar:
                if args.lowlight_FLAG:
                    # lowlight_param = random.uniform(-2, 0)
                    lowlight_param = 1
                    if random.randint(0, 2) > 0:
                        lowlight_param = random.uniform(1.5, 5)
                    _, summary, train_step_loss, train_step_loss_recovery, global_step_val = self.sess.run(
                        [train_op, self.write_op, self.loss, self.recovery_loss, self.global_step], feed_dict={
                            self.input_data: np.power(train_data[0], lowlight_param),# train_data[0]*np.exp(lowlight_param*np.log(2)),
                            self.label_sbbox: train_data[1],
                            self.label_mbbox: train_data[2],
                            self.label_lbbox: train_data[3],
                            self.true_sbboxes: train_data[4],
                            self.true_mbboxes: train_data[5],
                            self.true_lbboxes: train_data[6],
                            self.input_data_clean: train_data[0],
                            self.trainable: True,
                        })
                else:
                    if args.grayimage_FLAG:
                        bn,h,w,t  = train_data[0].shape
                        input_image = np.empty([bn,h,w,1],dtype=np.float64)
                        for bi in range(bn):
                            r,g,b = [train_data[0][bi,:,:,i] for i in range(3)]
                            img_gray = r*0.299+g*0.587+b*0.114
                            input_image[bi,:,:,0] = img_gray
                        _, summary, train_step_loss, global_step_val = self.sess.run(
                            [train_op, self.write_op, self.loss, self.global_step], feed_dict={
                                self.input_data: input_image,
                                self.label_sbbox: train_data[1],
                                self.label_mbbox: train_data[2],
                                self.label_lbbox: train_data[3],
                                self.true_sbboxes: train_data[4],
                                self.true_mbboxes: train_data[5],
                                self.true_lbboxes: train_data[6],
                                self.input_data_clean: train_data[0],
                                self.trainable: True,
                            })
                    else:
                        self.brightness_aver = train_data[7]
                        _, summary, train_step_loss, global_step_val = self.sess.run(
                            [train_op, self.write_op, self.loss, self.global_step], feed_dict={
                                self.input_data: train_data[0],
                                self.label_sbbox: train_data[1],
                                self.label_mbbox: train_data[2],
                                self.label_lbbox: train_data[3],
                                self.true_sbboxes: train_data[4],
                                self.true_mbboxes: train_data[5],
                                self.true_lbboxes: train_data[6],
                                self.input_data_clean: train_data[0],
                                self.trainable: True,
                            })

                train_epoch_loss.append(train_step_loss)
                self.summary_writer.add_summary(summary, global_step_val)

                pbar.set_description("train loss: %.2f"%(train_step_loss))

            if args.lowlight_FLAG:
                for test_data in self.testset:
                    # lowlight_param = random.uniform(-2, 0)
                    lowlight_param = 1
                    if random.randint(0, 2) > 0:
                        lowlight_param = random.uniform(1.5, 5)
                    test_step_loss = self.sess.run(self.loss, feed_dict={
                        self.input_data: np.power(test_data[0], lowlight_param), #test_data[0]*np.exp(lowlight_param*np.log(2)),
                        self.label_sbbox: test_data[1],
                        self.label_mbbox: test_data[2],
                        self.label_lbbox: test_data[3],
                        self.true_sbboxes: test_data[4],
                        self.true_mbboxes: test_data[5],
                        self.true_lbboxes: test_data[6],
                        self.input_data_clean: test_data[0],
                        self.trainable: False,
                    })

                    test_epoch_loss.append(test_step_loss)
            else:
                for test_data in self.testset:
                    test_step_loss = self.sess.run(self.loss, feed_dict={
                        self.input_data: test_data[0],
                        self.label_sbbox: test_data[1],
                        self.label_mbbox: test_data[2],
                        self.label_lbbox: test_data[3],
                        self.true_sbboxes: test_data[4],
                        self.true_mbboxes: test_data[5],
                        self.true_lbboxes: test_data[6],
                        self.input_data_clean: test_data[0],
                        self.trainable: False,
                    })

                    test_epoch_loss.append(test_step_loss)

            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            self.test_loss = test_epoch_loss
            train_epoch_loss_plot.append(train_epoch_loss)
            test_epoch_loss_plot.append(test_epoch_loss)
            ckpt_file = args.ckpt_dir + "/yolov3_test_loss=%.4f.ckpt" % test_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                            %(epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
            self.saver.save(self.sess, ckpt_file, global_step=epoch)

        plt.plot(range(0, 1+self.first_stage_epochs+self.second_stage_epochs),train_epoch_loss_plot,label='train')
        plt.plot(range(0, 1+self.first_stage_epochs+self.second_stage_epochs),test_epoch_loss_plot,label='test')
        fig = plt.gcf() # gcf - get current figure
        plt.title('Train and test loss in every epoch')# set plot title
        plt.xlabel('epoch')# set axis titles
        plt.ylabel('loss')
        fig.savefig(args.exp_dir + "/exp_" + args.exp_num + "/Loss.png")
        plt.cla() # clear axes for next plot

        
if __name__ == '__main__': YoloTrain().train()




