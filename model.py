
from utils import *
import os
import random
import tensorflow as tf
import numpy as np
from object_detection.utils import config_util
from object_detection.builders import model_builder


class resnet50:
    def __init__(self, gt_list=None, images_list_tensor=None, gt_tensor=None, gt_one_hot_tensor=None, category_idx=None, images_list=None, args=None):
        self.args = args
        self.category_index = category_idx
        self.model = self.load()
        self.images_list = images_list
        self.gt_list = gt_list
        self.images_list_tensor = images_list_tensor
        self.gt_tensor = gt_tensor
        self.gt_one_hot_tensor = gt_one_hot_tensor
        self.training_loop()
        self.inference()

    def load(self):
        tf.keras.backend.clear_session()

        print('Building model and restoring weights for fine-tuning...', flush=True)
        num_classes = 1
        pipeline_config = 'models/research/object_detection/configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config'
        checkpoint_path = 'models/research/object_detection/test_data/checkpoint/ckpt-0'

        # Load pipeline config and build a detection model.
        #
        # Since we are working off of a COCO architecture which predicts 90
        # class slots by default, we override the `num_classes` field here to be just
        # one (for our new rubber ducky class).
        configs = config_util.get_configs_from_pipeline_file(pipeline_config)
        model_config = configs['model']
        model_config.ssd.num_classes = num_classes
        model_config.ssd.freeze_batchnorm = True
        detection_model = model_builder.build(
            model_config=model_config, is_training=True)

        # Set up object-based checkpoint restore --- RetinaNet has two prediction
        # `heads` --- one for classification, the other for box regression.  We will
        # restore the box regression head but initialize the classification head
        # from scratch (we show the omission below by commenting out the line that
        # we would add if we wanted to restore both heads)
        fake_box_predictor = tf.compat.v2.train.Checkpoint(
            _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
            # _prediction_heads=detection_model._box_predictor._prediction_heads,
            #    (i.e., the classification head that we *will not* restore)
            _box_prediction_head=detection_model._box_predictor._box_prediction_head,
        )
        fake_model = tf.compat.v2.train.Checkpoint(
            _feature_extractor=detection_model._feature_extractor,
            _box_predictor=fake_box_predictor)
        ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
        ckpt.restore(checkpoint_path).expect_partial()

        # Run model through a dummy image so that variables are created
        image, shapes = detection_model.preprocess(tf.zeros([1, 640, 640, 3]))
        prediction_dict = detection_model.predict(image, shapes)
        _ = detection_model.postprocess(prediction_dict, shapes)
        print('Weights restored!')
        return detection_model

    def get_model_train_step_function(self, model, optimizer, vars_to_fine_tune, batch_size):
        """Get a tf.function for training step."""

        # Use tf.function for a bit of speed.
        # Comment out the tf.function decorator if you want the inside of the
        # function to run eagerly.
        @tf.function
        def train_step_fn(image_tensors,
                          groundtruth_boxes_list,
                          groundtruth_classes_list):
            """A single training iteration.

            Args:
              image_tensors: A list of [1, height, width, 3] Tensor of type tf.float32.
                Note that the height and width can vary across images, as they are
                reshaped within this function to be 640x640.
              groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
                tf.float32 representing groundtruth boxes for each image in the batch.
              groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
                with type tf.float32 representing groundtruth boxes for each image in
                the batch.

            Returns:
              A scalar tensor representing the total loss for the input batch.
            """
            shapes = tf.constant(batch_size * [[640, 640, 3]], dtype=tf.int32)
            model.provide_groundtruth(
                groundtruth_boxes_list=groundtruth_boxes_list,
                groundtruth_classes_list=groundtruth_classes_list)
            with tf.GradientTape() as tape:
                preprocessed_images = tf.concat(
                    [self.model.preprocess(image_tensor)[0]
                     for image_tensor in image_tensors], axis=0)
                prediction_dict = model.predict(preprocessed_images, shapes)
                losses_dict = model.loss(prediction_dict, shapes)
                total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
                gradients = tape.gradient(total_loss, vars_to_fine_tune)
                optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
            return total_loss

        return train_step_fn

    def training_loop(self):
        tf.keras.backend.set_learning_phase(True)

        # These parameters can be tuned; since our training set has 5 images
        # it doesn't make sense to have a much larger batch size, though we could
        # fit more examples in memory if we wanted to.
        batch_size = 8
        learning_rate = 0.01
        num_batches = 250

        # Select variables in top layers to fine-tune.
        trainable_variables = self.model.trainable_variables
        to_fine_tune = []
        prefixes_to_train = [
            'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
            'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead']
        for var in trainable_variables:
            if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
                to_fine_tune.append(var)

        # Set up forward + backward pass for a single train step.


        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        train_step_fn = self.get_model_train_step_function(self.model, optimizer, to_fine_tune, batch_size)

        print('Start fine-tuning!', flush=True)
        print('number of samples: ', len(self.images_list_tensor))
        for idx in range(num_batches):
            # Grab keys for a random subset of examples
            all_keys = list(range(len(self.images_list_tensor)))
            random.shuffle(all_keys)
            example_keys = all_keys[:batch_size]

            # Note that we do not do data augmentation in this demo.  If you want a
            # a fun exercise, we recommend experimenting with random horizontal flipping
            # and random cropping :)
            gt_boxes_list = [self.gt_tensor[key] for key in example_keys]
            gt_classes_list = [self.gt_one_hot_tensor[key] for key in example_keys]
            image_tensors = [self.images_list_tensor[key] for key in example_keys]

            # Training step (forward pass + backwards pass)
            total_loss = train_step_fn(image_tensors, gt_boxes_list, gt_classes_list)

            if idx % 10 == 0:
                print('batch ' + str(idx) + ' of ' + str(num_batches)
                      + ', loss=' + str(total_loss.numpy()), flush=True)

        print('Done fine-tuning!')

    def inference(self, image=None):
        if image is None:
            test_image_dir = self.args['test_image_dir']
            test_images_np = []
            for image_path in image_pre_load.imread_from_folder(test_image_dir):
                # image_path = os.path.join(test_image_dir, 'out' + str(i) + '.jpg')
                img = image_pre_load.imread(image_path)
                img = cv2.resize(img, (self.args['image_size'], self.args['image_size']))
                test_images_np.append(np.expand_dims(
                    img, axis=0))

            # Again, uncomment this decorator if you want to run inference eagerly
            @tf.function
            def detect(input_tensor):
                """Run detection on an input image.

                Args:
                  input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
                    Note that height and width can be anything since the image will be
                    immediately resized according to the needs of the model within this
                    function.

                Returns:
                  A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
                    and `detection_scores`).
                """
                preprocessed_image, shapes = self.model.preprocess(input_tensor)
                prediction_dict = self.model.predict(preprocessed_image, shapes)
                return self.model.postprocess(prediction_dict, shapes)

            # Note that the first frame will trigger tracing of the tf.function, which will
            # take some time, after which inference should be fast.

            label_id_offset = 1
            for i in range(len(test_images_np)):
                input_tensor = tf.convert_to_tensor(test_images_np[i], dtype=tf.float32)
                detections = detect(input_tensor)

                print('idx', i)
                print('detection_boxes: ', detections['detection_boxes'][0].numpy())
                print('detection_boxes: ', detections['detection_boxes'][0].numpy().shape)
                print('detection_classes: ', detections['detection_classes'][0].numpy())
                print('detection_classes: ', detections['detection_classes'][0].numpy().shape)
                print('detection_scores: ', detections['detection_scores'][0].numpy())
                print('detection_scores: ', detections['detection_scores'][0].numpy().shape)

                plot_detections(
                    test_images_np[i][0],
                    detections['detection_boxes'][0].numpy(),
                    detections['detection_classes'][0].numpy().astype(np.uint32)
                    + label_id_offset,
                    detections['detection_scores'][0].numpy(),
                    self.category_index, figsize=(15, 20), image_name="gif_frame_" + ('%02d' % i) + ".jpg")


class resnet101:
    def __init__(self, args=None):
        self.args = args
        self.model = None
        self.category_index = None
        # self.model = self.load()
        self.images_list = None
        self.gt_list = None
        self.images_list_tensor = None
        self.gt_tensor = None
        self.gt_one_hot_tensor = None

        # self.training_loop()
        # self.inference()

    # def load_train_loop(self):
    #     strategy = tf.compat.v2.distribute.MirroredStrategy()
    #     with strategy.scope():
    #         model_lib_v2.train_loop(
    #             pipeline_config_path=self.args['model_config'],
    #             model_dir=self.args['save_path'],
    #             train_steps=self.args['iteration'],
    #             use_tpu=False,
    #             checkpoint_every_n=50)

    def load(self):
        tf.keras.backend.clear_session()

        print('Building model and restoring weights for fine-tuning...', flush=True)
        num_classes = 1
        pipeline_config = self.args['model_config']
        checkpoint_path = self.args['model_checkpoint']

        # Load pipeline config and build a detection model.
        #
        # Since we are working off of a COCO architecture which predicts 90
        # class slots by default, we override the `num_classes` field here to be just
        # one (for our new rubber ducky class).
        configs = config_util.get_configs_from_pipeline_file(pipeline_config)
        model_config = configs['model']
        model_config.ssd.num_classes = num_classes
        model_config.ssd.freeze_batchnorm = True
        detection_model = model_builder.build(
            model_config=model_config, is_training=True)

        # Set up object-based checkpoint restore --- RetinaNet has two prediction
        # `heads` --- one for classification, the other for box regression.  We will
        # restore the box regression head but initialize the classification head
        # from scratch (we show the omission below by commenting out the line that
        # we would add if we wanted to restore both heads)
        fake_box_predictor = tf.compat.v2.train.Checkpoint(
            _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
            # _prediction_heads=detection_model._box_predictor._prediction_heads,
            #    (i.e., the classification head that we *will not* restore)
            _box_prediction_head=detection_model._box_predictor._box_prediction_head,
        )
        fake_model = tf.compat.v2.train.Checkpoint(
            _feature_extractor=detection_model._feature_extractor,
            _box_predictor=fake_box_predictor)
        ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
        ckpt.restore(checkpoint_path).expect_partial()

        # Run model through a dummy image so that variables are created
        image, shapes = detection_model.preprocess(tf.zeros([1, self.args['image_size'], self.args['image_size'], 3]))
        prediction_dict = detection_model.predict(image, shapes)
        _ = detection_model.postprocess(prediction_dict, shapes)
        print('Weights restored!')
        return detection_model

    def get_model_train_step_function(self, model, optimizer, vars_to_fine_tune, batch_size):
        """Get a tf.function for training step."""

        # Use tf.function for a bit of speed.
        # Comment out the tf.function decorator if you want the inside of the
        # function to run eagerly.
        # @tf.function
        def train_step_fn(image_tensors,
                          groundtruth_boxes_list,
                          groundtruth_classes_list):
            """A single training iteration.

            Args:
              image_tensors: A list of [1, height, width, 3] Tensor of type tf.float32.
                Note that the height and width can vary across images, as they are
                reshaped within this function to be 640x640.
              groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
                tf.float32 representing groundtruth boxes for each image in the batch.
              groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
                with type tf.float32 representing groundtruth boxes for each image in
                the batch.

            Returns:
              A scalar tensor representing the total loss for the input batch.
            """
            shapes = tf.constant(batch_size * [[640, 640, 3]], dtype=tf.int32)
            model.provide_groundtruth(
                groundtruth_boxes_list=groundtruth_boxes_list,
                groundtruth_classes_list=groundtruth_classes_list)
            with tf.GradientTape() as tape:
                preprocessed_images = tf.concat(
                    [self.model.preprocess(image_tensor)[0]
                     for image_tensor in image_tensors], axis=0)
                prediction_dict = model.predict(preprocessed_images, shapes)
                losses_dict = model.loss(prediction_dict, shapes)
                total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
                gradients = tape.gradient(total_loss, vars_to_fine_tune)
                optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))

            return total_loss

        return train_step_fn

    def training_loop(self, gt_list=None, images_list_tensor=None, gt_tensor=None, gt_one_hot_tensor=None, category_idx=None, images_list=None,):
        self.category_index = category_idx
        # self.model = self.load()
        self.images_list = images_list
        self.gt_list = gt_list
        self.images_list_tensor = images_list_tensor
        self.gt_tensor = gt_tensor
        self.gt_one_hot_tensor = gt_one_hot_tensor

        tf.keras.backend.set_learning_phase(True)

        # These parameters can be tuned; since our training set has 5 images
        # it doesn't make sense to have a much larger batch size, though we could
        # fit more examples in memory if we wanted to.
        batch_size = self.args['batch_size']
        learning_rate = self.args['learning_rate']
        num_batches = self.args['iteration']

        # Select variables in top layers to fine-tune.
        trainable_variables = self.model.trainable_variables
        to_fine_tune = []
        prefixes_to_train = [
            'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
            'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead']
        for var in trainable_variables:
            if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
                to_fine_tune.append(var)

        # Set up forward + backward pass for a single train step.


        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        train_step_fn = self.get_model_train_step_function(self.model, optimizer, to_fine_tune, batch_size)

        print('Start fine-tuning!', flush=True)
        print('number of samples: ', len(self.images_list_tensor))
        for idx in range(num_batches):
            # Grab keys for a random subset of examples
            all_keys = list(range(len(self.images_list_tensor)))
            random.shuffle(all_keys)
            example_keys = all_keys[:batch_size]

            # Note that we do not do data augmentation in this demo.  If you want a
            # a fun exercise, we recommend experimenting with random horizontal flipping
            # and random cropping :)
            gt_boxes_list = [self.gt_tensor[key] for key in example_keys]
            gt_classes_list = [self.gt_one_hot_tensor[key] for key in example_keys]
            image_tensors = [self.images_list_tensor[key] for key in example_keys]

            # Training step (forward pass + backwards pass)
            total_loss = train_step_fn(image_tensors, gt_boxes_list, gt_classes_list)

            if idx % 10 == 0:
                print('batch ' + str(idx) + ' of ' + str(num_batches)
                      + ', loss=' + str(total_loss.numpy()), flush=True)

        print('Done fine-tuning!')

    def inference(self, image=None):

        names_dict={}
        if image is None:
            test_image_dir = self.args['test_image_dir']
            test_images_np = []
            if self.model is None:
                self.load_saved()

            for i, imagePath in enumerate(image_pre_load.imread_from_folder(os.path.join(test_image_dir))):
                # image_path = os.path.join(test_image_dir, 'out' + str(i) + '.jpg')
                im = image_pre_load.imread(imagePath)
                im = cv2.resize(im, (self.args['image_size'], self.args['image_size']))
                names_dict[i] = os.path.basename(imagePath)
                test_images_np.append(np.expand_dims(im, axis=0))

            # Again, uncomment this decorator if you want to run inference eagerly
            @tf.function
            def detect(input_tensor):
                """Run detection on an input image.

                Args:
                  input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
                    Note that height and width can be anything since the image will be
                    immediately resized according to the needs of the model within this
                    function.

                Returns:
                  A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
                    and `detection_scores`).
                """

                preprocessed_image, shapes = self.model.preprocess(input_tensor)
                prediction_dict = self.model.predict(preprocessed_image, shapes)
                return self.model.postprocess(prediction_dict, shapes)

            # Note that the first frame will trigger tracing of the tf.function, which will
            # take some time, after which inference should be fast.

            detections_dict = {}
            label_id_offset = 1
            for i in range(len(test_images_np)):
                input_tensor = tf.convert_to_tensor(test_images_np[i], dtype=tf.float32)
                detections = detect(input_tensor)

                detections_dict[names_dict[i]] = detections

                # print('idx', i)
                # print('detection_boxes: ', detections['detection_boxes'][0].numpy())
                # print('detection_boxes: ', detections['detection_boxes'][0].numpy().shape)
                # print('detection_classes: ', detections['detection_classes'][0].numpy())
                # print('detection_classes: ', detections['detection_classes'][0].numpy().shape)
                # print('detection_scores: ', detections['detection_scores'][0].numpy())
                # print('detection_scores: ', detections['detection_scores'][0].numpy().shape)

                if self.args['show_test_samples']:
                    if self.category_index is None:
                        self.category_index = {'1': 'flowers'}
                    plot_detections(
                        test_images_np[i][0],
                        detections['detection_boxes'][0].numpy(),
                        detections['detection_classes'][0].numpy().astype(np.uint32)
                        + label_id_offset,
                        detections['detection_scores'][0].numpy(),
                        self.category_index, figsize=(15, 20), image_name="gif_frame_" + ('%02d' % i) + ".jpg")

            return detections_dict

    def get_filepath(self, strategy, filepath):
        """Get appropriate filepath for worker.

        Args:
          strategy: A tf.distribute.Strategy object.
          filepath: A path to where the Checkpoint object is stored.

        Returns:
          A temporary filepath for non-chief workers to use or the original filepath
          for the chief.
        """
        if strategy.extended.should_checkpoint:
            return filepath
        else:
            # TODO(vighneshb) Replace with the public API when TF exposes it.
            task_id = strategy.extended._task_id  # pylint:disable=protected-access
            return os.path.join(filepath, 'temp_worker_{:03d}'.format(task_id))

    def save(self):

        # strategy = tf.compat.v2.distribute.get_strategy()
        # summary_writer_filepath = self.get_filepath(strategy,
        #                                        os.path.join(self.args['save_path'], 'train'))
        # record_summaries = True
        # if record_summaries:
        #     summary_writer = tf.compat.v2.summary.create_file_writer(
        #         summary_writer_filepath)
        # else:
        #     summary_writer = tf.summary.create_noop_writer()

        # tf.compat.v1.disable_eager_execution()
        path = self.args['save_path']
        print('Saving the Model ... ')
        dir_path = os.path.join(path, self.category_index[self.args['class_id']]['name'], 'ckpt')

        fake_box_predictor = tf.compat.v2.train.Checkpoint(
            _base_tower_layers_for_heads=self.model._box_predictor._base_tower_layers_for_heads,
            _prediction_heads=self.model._box_predictor._prediction_heads,
            _box_prediction_head=self.model._box_predictor._box_prediction_head,
        )
        fake_model = tf.compat.v2.train.Checkpoint(
            _feature_extractor=self.model._feature_extractor,
            _box_predictor=fake_box_predictor)
        ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
        ckpt.save(dir_path)
        self.model = None

        # with summary_writer.as_default():
        #     with strategy.scope():
        #         with tf.compat.v2.summary.record_if(
        #                 lambda: global_step % num_steps_per_iteration == 0):
        #             # Load a fine-tuning checkpoint.
        #             if train_config.fine_tune_checkpoint:
        #                 load_fine_tune_checkpoint(detection_model,
        #                                           train_config.fine_tune_checkpoint,
        #                                           fine_tune_checkpoint_type,
        #                                           fine_tune_checkpoint_version,
        #                                           train_input,
        #                                           unpad_groundtruth_tensors)
        #
        #             ckpt = tf.compat.v2.train.Checkpoint(
        #                 step=global_step, model=detection_model, optimizer=optimizer)
        #
        #             manager_dir = get_filepath(strategy, model_dir)
        #             if not strategy.extended.should_checkpoint:
        #                 checkpoint_max_to_keep = 1
        #             manager = tf.compat.v2.train.CheckpointManager(
        #                 ckpt, manager_dir, max_to_keep=checkpoint_max_to_keep)
        #
        #             # We use the following instead of manager.latest_checkpoint because
        #             # manager_dir does not point to the model directory when we are running
        #             # in a worker.
        #             latest_checkpoint = tf.train.latest_checkpoint(model_dir)
        #             ckpt.restore(latest_checkpoint)
        #
        #             def train_step_fn(features, labels):
        #                 """Single train step."""
        #                 loss = eager_train_step(
        #                     detection_model,
        #                     features,
        #                     labels,
        #                     unpad_groundtruth_tensors,
        #                     optimizer,
        #                     learning_rate=learning_rate_fn(),
        #                     add_regularization_loss=add_regularization_loss,
        #                     clip_gradients_value=clip_gradients_value,
        #                     global_step=global_step,
        #                     num_replicas=strategy.num_replicas_in_sync)
        #                 global_step.assign_add(1)
        #                 return loss
        #
        #             def _sample_and_train(strategy, train_step_fn, data_iterator):
        #                 features, labels = data_iterator.next()
        #                 if hasattr(tf.distribute.Strategy, 'run'):
        #                     per_replica_losses = strategy.run(
        #                         train_step_fn, args=(features, labels))
        #                 else:
        #                     per_replica_losses = strategy.experimental_run_v2(
        #                         train_step_fn, args=(features, labels))
        #                 # TODO(anjalisridhar): explore if it is safe to remove the
        #                 ## num_replicas scaling of the loss and switch this to a ReduceOp.Mean
        #                 return strategy.reduce(tf.distribute.ReduceOp.SUM,
        #                                        per_replica_losses, axis=None)
        #
        #             @tf.function
        #             def _dist_train_step(data_iterator):
        #                 """A distributed train step."""
        #
        #                 if num_steps_per_iteration > 1:
        #                     for _ in tf.range(num_steps_per_iteration - 1):
        #                         # Following suggestion on yaqs/5402607292645376
        #                         with tf.name_scope(''):
        #                             _sample_and_train(strategy, train_step_fn, data_iterator)
        #
        #                 return _sample_and_train(strategy, train_step_fn, data_iterator)
        #
        #             train_input_iter = iter(train_input)
        #
        #             if int(global_step.value()) == 0:
        #                 manager.save()
        #
        #             checkpointed_step = int(global_step.value())
        #             logged_step = global_step.value()
        #
        #             last_step_time = time.time()
        #             for _ in range(global_step.value(), train_steps,
        #                            num_steps_per_iteration):
        #
        #                 loss = _dist_train_step(train_input_iter)
        #
        #                 time_taken = time.time() - last_step_time
        #                 last_step_time = time.time()
        #                 steps_per_sec = num_steps_per_iteration * 1.0 / time_taken
        #
        #                 tf.compat.v2.summary.scalar(
        #                     'steps_per_sec', steps_per_sec, step=global_step)
        #
        #                 steps_per_sec_list.append(steps_per_sec)
        #
        #                 if global_step.value() - logged_step >= 100:
        #                     tf.logging.info(
        #                         'Step {} per-step time {:.3f}s loss={:.3f}'.format(
        #                             global_step.value(), time_taken / num_steps_per_iteration,
        #                             loss))
        #                     logged_step = global_step.value()
        #
        #                 if ((int(global_step.value()) - checkpointed_step) >=
        #                         checkpoint_every_n):
        #                     manager.save()
        #                     checkpointed_step = int(global_step.value())

    def load_saved(self):

        if self.model is None:

            # print('Building model and restoring weights for fine-tuning...', flush=True)
            num_classes = 1
            pipeline_config = self.args['model_config']

            configs = config_util.get_configs_from_pipeline_file(pipeline_config)
            model_config = configs['model']
            model_config.ssd.num_classes = num_classes
            model_config.ssd.freeze_batchnorm = True
            detection_model = model_builder.build(
                model_config=model_config, is_training=True)
            path = self.args['load_path']
            print('Loading the Model ... ')
            dir_path = self.args['save_path'] + '\\' + self.args['object_name'] + '\\ckpt-1' #E:\\Programming\\Multi_object_detection\\saved_chckpoints\\flowers\\ckpt-1'
            fake_box_predictor = tf.compat.v2.train.Checkpoint(
                _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
                _prediction_heads=detection_model._box_predictor._prediction_heads,
                _box_prediction_head=detection_model._box_predictor._box_prediction_head,
            )
            fake_model = tf.compat.v2.train.Checkpoint(
                _feature_extractor=detection_model._feature_extractor,
                _box_predictor=fake_box_predictor)


            ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
            ckpt.restore(dir_path).expect_partial()
            image, shapes = detection_model.preprocess(tf.zeros([1, self.args['image_size'], self.args['image_size'], 3]))
            prediction_dict = detection_model.predict(image, shapes)
            _ = detection_model.postprocess(prediction_dict, shapes)
            print('Weights restored!')
            self.model = detection_model



