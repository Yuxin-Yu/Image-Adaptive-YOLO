#! /usr/bin/env python
# coding=utf-8


import numpy as np
import tensorflow as tf
import core.utils as utils
import core.common as common
import core.backbone as backbone
from core.config_lowlight import cfg
from core.config_lowlight import args
from mymodule.modules import *

class YOLOV3(object):
    """Implement tensoflow yolov3 here"""
    def __init__(self, input_data, trainable, input_data_clean,brightness_aver):

        self.trainable        = trainable
        self.classes          = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_class        = len(self.classes)
        self.strides          = np.array(cfg.YOLO.STRIDES)
        self.anchors          = utils.get_anchors(cfg.YOLO.ANCHORS)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.iou_loss_thresh  = cfg.YOLO.IOU_LOSS_THRESH
        self.upsample_method  = cfg.YOLO.UPSAMPLE_METHOD
        self.isp_flag = cfg.YOLO.ISP_FLAG
        self.end_points = {}


        try:
            self.conv_lbbox, self.conv_mbbox, self.conv_sbbox, self.recovery_loss,self.end_points = self.__build_nework(input_data, self.isp_flag, input_data_clean,brightness_aver)
        except:
            raise NotImplementedError("Can not build up yolov3 network!")

        with tf.compat.v1.variable_scope('pred_sbbox'):
            self.pred_sbbox = self.decode(self.conv_sbbox, self.anchors[0], self.strides[0])

        with tf.compat.v1.variable_scope('pred_mbbox'):
            self.pred_mbbox = self.decode(self.conv_mbbox, self.anchors[1], self.strides[1])

        with tf.compat.v1.variable_scope('pred_lbbox'):
            self.pred_lbbox = self.decode(self.conv_lbbox, self.anchors[2], self.strides[2])

    

    def __build_nework(self, input_data, isp_flag, input_data_clean,brightness_aver):

        filtered_image_batch = input_data
        self.filter_params = input_data
        filter_imgs_series = []
        end_points = {}

        if isp_flag:
            if args.filter_bank_FLAG:
                with tf.compat.v1.variable_scope('extract_parameters'):
                    def emu_night_parameter(input_data,filtered_image_batch,cfg,trainable):
                        filter_features_night = common.extract_parameters_2(input_data, cfg,trainable)
                        filters = cfg.filters
                        filters = [x(input_data, cfg) for x in filters]
                        filter_parameters = []
                        for j, filter in enumerate(filters):
                            with tf.compat.v1.variable_scope('filter_%d' % j):
                                print('    creating night filter:', j, 'name:', str(filter.__class__), 'abbr.',
                                    filter.get_short_name())
                                print('      filter_features:', filter_features_night.shape)

                                filtered_image_batch, filter_parameter = filter.apply(
                                    filtered_image_batch, filter_features_night)
                                filter_parameters.append(filter_parameter)
                                print('      output:', filtered_image_batch.shape)
                        #self.filter_params_night = filter_parameters
                        return filtered_image_batch
                    
                    def Nonefun(filtered_image_batch):
                        return filtered_image_batch
                    def emu_explosive_parameter(input_data,filtered_image_batch,cfg,trainable):
                        filter_features_explosive = common.extract_parameters_3(input_data, cfg,trainable)
                        filters = cfg.filters
                        filters = [x(input_data, cfg) for x in filters]
                        filter_parameters = []
                        for j, filter in enumerate(filters):
                            with tf.compat.v1.variable_scope('filter_%d' % j):
                                print('    creating explosive filter:', j, 'name:', str(filter.__class__), 'abbr.',
                                    filter.get_short_name())
                                print('      filter_features:', filter_features_explosive.shape)

                                filtered_image_batch, filter_parameter = filter.apply(
                                    filtered_image_batch, filter_features_explosive)
                                filter_parameters.append(filter_parameter)
                                print('      output:', filtered_image_batch.shape)
                        #self.filter_params_night = filter_parameters
                        return filtered_image_batch
                    def switch_parameter(input_data,filtered_image_batch,cfg,trainable,brightness_mean):
                        filtered_image_batch = tf.cond(brightness_mean <=168,lambda : Nonefun(filtered_image_batch),\
                            lambda : emu_explosive_parameter(input_data,filtered_image_batch,cfg,trainable))
                        return filtered_image_batch
                    #brightness_mean = np.mean(input_data)
                    brightness_mean = brightness_aver[0]
                    
                    input_data = tf.image.resize(input_data, [256, 256], method=tf.image.ResizeMethod.BILINEAR)
                    filtered_image_batch = tf.cond(brightness_mean < 90,lambda : emu_night_parameter(input_data,filtered_image_batch,cfg,self.trainable),\
                        lambda :switch_parameter(input_data,filtered_image_batch,cfg,self.trainable,brightness_mean))
                    '''
                    if brightness_mean < 90: #night
                        filter_features_night = common.extract_parameters_2(input_data, cfg, self.trainable)
                        filters = cfg.filters
                        filters = [x(input_data, cfg) for x in filters]
                        filter_parameters = []
                        for j, filter in enumerate(filters):
                            with tf.compat.v1.variable_scope('filter_%d' % j):
                                print('    creating night filter:', j, 'name:', str(filter.__class__), 'abbr.',
                                    filter.get_short_name())
                                print('      filter_features:', filter_features_night.shape)

                                filtered_image_batch, filter_parameter = filter.apply(
                                    filtered_image_batch, filter_features_night)
                                filter_parameters.append(filter_parameter)
                                filter_imgs_series.append(filtered_image_batch)


                                print('      output:', filtered_image_batch.shape)
                        self.filter_params = filter_parameters
                    elif brightness_mean > 168: #explosive
                        filter_features_explosive = common.extract_parameters_3(input_data, cfg, self.trainable)
                        filters = cfg.filters
                        filters = [x(input_data, cfg) for x in filters]
                        filter_parameters = []
                        for j, filter in enumerate(filters):
                            with tf.compat.v1.variable_scope('filter_%d' % j):
                                print('    creating explosive filter:', j, 'name:', str(filter.__class__), 'abbr.',
                                    filter.get_short_name())
                                print('      filter_features:', filter_features_explosive.shape)

                                filtered_image_batch, filter_parameter = filter.apply(
                                    filtered_image_batch, filter_features_explosive)
                                filter_parameters.append(filter_parameter)
                                filter_imgs_series.append(filtered_image_batch)

                                print('      output:', filtered_image_batch.shape)
                        self.filter_params = filter_parameters
                    else: #daytime
                        pass
                    '''
            else:

                with tf.compat.v1.variable_scope('extract_parameters_2'):
                    input_data = tf.image.resize(input_data, [256, 256], method=tf.image.ResizeMethod.BILINEAR)
                    filter_features = common.extract_parameters_2(input_data, cfg, self.trainable)

                # filter_features = tf.random_normal([1, 10], 0.5, 0.1)

                filters = cfg.filters
                filters = [x(input_data, cfg) for x in filters]
                filter_parameters = []
                for j, filter in enumerate(filters):
                    with tf.compat.v1.variable_scope('filter_%d' % j):
                        print('    creating filter:', j, 'name:', str(filter.__class__), 'abbr.',
                            filter.get_short_name())
                        print('      filter_features:', filter_features.shape)

                        filtered_image_batch, filter_parameter = filter.apply(
                            filtered_image_batch, filter_features)
                        filter_parameters.append(filter_parameter)
                        filter_imgs_series.append(filtered_image_batch)


                        print('      output:', filtered_image_batch.shape)
                self.filter_params = filter_parameters
        self.image_isped = filtered_image_batch
        self.filter_imgs_series = filter_imgs_series

        recovery_loss = tf.reduce_sum(input_tensor=tf.pow(filtered_image_batch - input_data_clean, 2.0))#/(2.0 * batch_size)

        input_data = filtered_image_batch
        route_1, route_2, input_data = backbone.darknet53(input_data, self.trainable)
        if args.aspp_FLAG_ser:
            input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'aspp_conv0')
            input_data = common.convolutional(input_data, (3, 3,  512, 1024), self.trainable, 'aspp_conv1')
            input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'aspp_conv2')

            aspp_list = ASPP(input_data, [6, 12, 18], 512)
            end_points["aspp1"] = aspp_list[0]
            end_points["aspp2"] = aspp_list[1]
            end_points["aspp3"] = aspp_list[2]
            end_points["aspp4"] = aspp_list[3]

            # Image Pooling
            with tf.compat.v1.variable_scope("img_pool"):
                # print("net:", net.shape)
                pooled = tf.reduce_mean(input_tensor=input_data, axis=[1, 2], name="avg_pool", keepdims=True)
                # end_points["aspp5"] = pooled

                global_feat = tf_slim.conv2d(pooled, num_outputs=512, kernel_size=1, stride=1, scope="conv1x1")
                global_feat = tf.image.resize(global_feat, tf.shape(input=input_data)[1:3], method=tf.image.ResizeMethod.BILINEAR)
                # print("global_feat:", global_feat.shape)
                aspp_list.append(global_feat)
                end_points["aspp5"] = global_feat

            input_data = tf.concat(aspp_list, axis=3)
            end_points['fusion'] = input_data
            input_data = common.convolutional(input_data, (1, 1, 2560,  1024), self.trainable, 'aspp_conv3')
            input_data = common.convolutional(input_data, (3, 3,  1024, 2048), self.trainable, 'aspp_conv4')
            input_data = common.convolutional(input_data, (1, 1, 2048,  1024), self.trainable, 'aspp_conv5')


        if args.aspp_FLAG:
            layer1=tf.compat.v1.layers.conv2d(input_data,512,3,strides=1, padding='same',dilation_rate=6)
            layer1=tf.compat.v1.layers.conv2d(layer1,512,1,strides=1, padding='same')
            layer1=tf.compat.v1.layers.conv2d(layer1,512,1,strides=1, padding='same')

        input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv52')
        input_data = common.convolutional(input_data, (3, 3,  512, 1024), self.trainable, 'conv53')
        input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv54')
        input_data = common.convolutional(input_data, (3, 3,  512, 1024), self.trainable, 'conv55')
        input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv56')
        
        if args.aspp_FLAG:
            input_data_fpn2 = input_data
            input_data = tf.concat([input_data,layer1], axis=-1)

        conv_lobj_branch = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, name='conv_lobj_branch')
        conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3*(self.num_class + 5)),
                                          trainable=self.trainable, name='conv_lbbox', activate=False, bn=False)
        if args.aspp_FLAG:
            input_data = common.convolutional(input_data_fpn2, (1, 1,  512,  256), self.trainable, 'conv57')
        else:
            input_data = common.convolutional(input_data, (1, 1,  512,  256), self.trainable, 'conv57')
        input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)

        with tf.compat.v1.variable_scope('route_1'):
            input_data = tf.concat([input_data, route_2], axis=-1)

        if args.aspp_FLAG:
            layer2=tf.compat.v1.layers.conv2d(input_data,256,3,strides=1, padding='same',dilation_rate=12)
            layer2=tf.compat.v1.layers.conv2d(layer2,256,1,strides=1, padding='same')
            layer2=tf.compat.v1.layers.conv2d(layer2,256,1,strides=1, padding='same')

        input_data = common.convolutional(input_data, (1, 1, 768, 256), self.trainable, 'conv58')
        input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv59')
        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv60')
        input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv61')
        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv62')

        if args.aspp_FLAG:
            input_data_fpn3 = input_data
            input_data = tf.concat([input_data,layer2], axis=-1)

        conv_mobj_branch = common.convolutional(input_data, (3, 3, 256, 512),  self.trainable, name='conv_mobj_branch' )
        conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3*(self.num_class + 5)),
                                          trainable=self.trainable, name='conv_mbbox', activate=False, bn=False)

        if args.aspp_FLAG:
            input_data = common.convolutional(input_data_fpn3, (1, 1, 256, 128), self.trainable, 'conv63')
        else:
            input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv63')
        input_data = common.upsample(input_data, name='upsample1', method=self.upsample_method)

        with tf.compat.v1.variable_scope('route_2'):
            input_data = tf.concat([input_data, route_1], axis=-1)

        if args.aspp_FLAG:
            layer3=tf.compat.v1.layers.conv2d(input_data,128,3,strides=1, padding='same',dilation_rate=18)
            layer3=tf.compat.v1.layers.conv2d(layer3,128,1,strides=1, padding='same')
            layer3=tf.compat.v1.layers.conv2d(layer3,128,1,strides=1, padding='same')

        input_data = common.convolutional(input_data, (1, 1, 384, 128), self.trainable, 'conv64')
        input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv65')
        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv66')
        input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv67')
        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv68')

        if args.aspp_FLAG:
            input_data = tf.concat([input_data,layer3], axis=-1)

        conv_sobj_branch = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, name='conv_sobj_branch')
        conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3*(self.num_class + 5)),
                                          trainable=self.trainable, name='conv_sbbox', activate=False, bn=False)

        return conv_lbbox, conv_mbbox, conv_sbbox, recovery_loss,end_points

    def decode(self, conv_output, anchors, stride):
        """
        return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
               contains (x, y, w, h, score, probability)
        """

        conv_shape       = tf.shape(input=conv_output)
        batch_size       = conv_shape[0]
        output_size      = conv_shape[1]
        anchor_per_scale = len(anchors)

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, anchor_per_scale, 5 + self.num_class))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5: ]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def focal(self, target, actual, alpha=1, gamma=2):
        focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    def bbox_giou(self, boxes1, boxes2):

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / union_area

        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou

    def bbox_iou(self, boxes1, boxes2):

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / union_area

        return iou

    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):

        conv_shape  = tf.shape(input=conv)
        batch_size  = conv_shape[0]
        output_size = conv_shape[1]
        input_size  = stride * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size,
                                 self.anchor_per_scale, 5 + self.num_class))
        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh     = pred[:, :, :, :, 0:4]
        pred_conf     = pred[:, :, :, :, 4:5]

        label_xywh    = label[:, :, :, :, 0:4]
        respond_bbox  = label[:, :, :, :, 4:5]
        label_prob    = label[:, :, :, :, 5:]

        giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(input_tensor=iou, axis=-1), axis=-1)

        respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < self.iou_loss_thresh, tf.float32 )

        conf_focal = self.focal(respond_bbox, pred_conf)

        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=giou_loss, axis=[1,2,3,4]))
        conf_loss = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=conf_loss, axis=[1,2,3,4]))
        prob_loss = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=prob_loss, axis=[1,2,3,4]))

        return giou_loss, conf_loss, prob_loss



    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):

        with tf.compat.v1.name_scope('smaller_box_loss'):
            loss_sbbox = self.loss_layer(self.conv_sbbox, self.pred_sbbox, label_sbbox, true_sbbox,
                                         anchors = self.anchors[0], stride = self.strides[0])

        with tf.compat.v1.name_scope('medium_box_loss'):
            loss_mbbox = self.loss_layer(self.conv_mbbox, self.pred_mbbox, label_mbbox, true_mbbox,
                                         anchors = self.anchors[1], stride = self.strides[1])

        with tf.compat.v1.name_scope('bigger_box_loss'):
            loss_lbbox = self.loss_layer(self.conv_lbbox, self.pred_lbbox, label_lbbox, true_lbbox,
                                         anchors = self.anchors[2], stride = self.strides[2])

        with tf.compat.v1.name_scope('giou_loss'):
            giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

        with tf.compat.v1.name_scope('conf_loss'):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

        with tf.compat.v1.name_scope('prob_loss'):
            prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]
        with tf.compat.v1.name_scope('recovery_loss'):
            recovery_loss = self.recovery_loss

        return giou_loss, conf_loss, prob_loss, recovery_loss

    def get_endpoints(self):
        return self.end_points


