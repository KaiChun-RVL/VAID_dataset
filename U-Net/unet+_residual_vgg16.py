from keras import regularizers
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau
import cv2
from keras import applications
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
from datagenerator import *
from keras.utils.vis_utils import plot_model
import lovasz_loss as lovasz
import keras

class myUnet(object):
    def __init__(self, img_rows=512, img_cols=512):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data()
        return imgs_train, imgs_mask_train, imgs_test
    

    def get_vgg16_unet(self):

        def swish(x):
            return (K.sigmoid(x) * x)
        
        def class_tversky(y_true, y_pred):
            smooth = 1

            y_true = K.permute_dimensions(y_true, (3,1,2,0))
            y_pred = K.permute_dimensions(y_pred, (3,1,2,0))

            y_true_pos = K.batch_flatten(y_true)
            y_pred_pos = K.batch_flatten(y_pred)
            true_pos = K.sum(y_true_pos * y_pred_pos, 1)
            false_neg = K.sum(y_true_pos * (1-y_pred_pos), 1)
            false_pos = K.sum((1-y_true_pos)*y_pred_pos, 1)
            alpha = 0.3
            return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

        def focal_tversky_loss(y_true,y_pred):
            pt_1 = class_tversky(y_true, y_pred)
            gamma = 1.33
            return K.sum(K.pow((1-pt_1), gamma))
                
        def generalized_dice_coeff(y_true, y_pred):
            Ncl = y_pred.shape[-1]
            w = K.zeros(shape=(Ncl,))
            w = K.sum(y_true, axis=(0,1,2))
            w = 1/(w**2+0.000001)
            # Compute gen dice coef:
            numerator = y_true*y_pred
            numerator = w*K.sum(numerator,(0,1,2,3))
            numerator = K.sum(numerator)

            denominator = y_true+y_pred
            denominator = w*K.sum(denominator,(0,1,2,3))
            denominator = K.sum(denominator)

            gen_dice_coef = 2*numerator/denominator

            return gen_dice_coef

        def generalized_dice_loss(y_true, y_pred):
            return 1 - generalized_dice_coeff(y_true, y_pred)

            
        deep_supervision = False
        activation_method = swish
        dropout_rate = 0.3
            
        
        def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):

            branch_0 = Conv2D(32, (1, 1), activation=activation_method, kernel_initializer = 'he_normal', padding='same', kernel_regularizer=regularizers.l2(1e-4))(input_tensor)
            
            branch_1 = Conv2D(32, (1, 1), activation=activation_method, kernel_initializer = 'he_normal', padding='same', kernel_regularizer=regularizers.l2(1e-4))(input_tensor)
            branch_1 = Conv2D(32, (3, 3), activation=activation_method, kernel_initializer = 'he_normal', padding='same', kernel_regularizer=regularizers.l2(1e-4))(branch_1)

            branch_2 = Conv2D(32, (1, 1), activation=activation_method, kernel_initializer = 'he_normal', padding='same', kernel_regularizer=regularizers.l2(1e-4))(input_tensor)
            branch_2 = Conv2D(48, (3, 3), activation=activation_method, kernel_initializer = 'he_normal', padding='same', kernel_regularizer=regularizers.l2(1e-4))(branch_2)
            branch_2 = Conv2D(64, (3, 3), activation=activation_method, kernel_initializer = 'he_normal', padding='same', kernel_regularizer=regularizers.l2(1e-4))(branch_2)
            
            main = concatenate([branch_0, branch_1, branch_2], axis=3)
            main = Conv2D(K.int_shape(input_tensor)[3], kernel_size = (1, 1), activation=None, use_bias = True, padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4))(main)

            main = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,  #residual
               output_shape=K.int_shape(input_tensor)[1:],
               arguments={'scale': scale},
               name='residual' + stage)([input_tensor, main])
            main = Activation(activation_method)(main)
            return main
        
        def standard_conv(input_tensor, stage, nb_filter, kernel_size=3):

            x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=activation_method, name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=regularizers.l2(1e-4))(input_tensor)
            #x = BatchNormalization()(x)
            #x = Dropout(dropout_rate, name='dp'+stage+'_1')(x)
            x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=activation_method, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
            #x = BatchNormalization()(x)
            #x = Dropout(dropout_rate, name='dp'+stage+'_2')(x)

            return x

        def shortcut(input_tensor, shortcut_tensor, nb_filter, stage, kernel_size=1):
            x = Conv2D(nb_filter, kernel_size = (1, 1), name = 'shortcut' + stage, padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4), strides=1)(shortcut_tensor)
            #x = BatchNormalization()(x)
            input_tensor = add([x, input_tensor])
            input_tensor = Activation(activation_method)(input_tensor)
            return input_tensor
        

        
        def conv_block_simple(prevlayer, filters, stage, strides=(1, 1)):
            conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=stage + "_conv")(prevlayer)
            conv = BatchNormalization(name=stage + "_bn")(conv)
            conv = Activation(swish, name=stage + "_activation")(conv)
            
            conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=stage + "_conv2")(conv)
            conv = BatchNormalization(name=stage + "_bn2")(conv)
            conv = Activation(swish, name=stage + "_activation2")(conv)
            return conv
        
        
        inputs = Input((self.img_rows, self.img_cols, 3))


        # If you want to specify input tensor shape, e.g. 256x256 with 3 channels:
        vgg_model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
        #print([layer.name for layer in vgg_model.layers])
        layers = dict([(layer.name, layer) for layer in vgg_model.layers])
        
        

        # Now getting bottom layers for multi-scale skip-layers
        #block1_conv2 = layers['block1_conv2'].output       #(None, 512, 512, 64)
        #block2_conv2 = layers['block2_conv2'].output      #(None, 256, 256, 128
        #block3_conv3 = layers['block3_conv3'].output      #(None, 128, 128, 256
       # block4_conv3 = layers['block4_conv3'].output      #(None, 64, 64, 512)
        #vgg_top = layers['block5_conv3'].output           #(None, 32, 32, 512)
      
#----------------------------------------------------------------------------------------------------------------------------------------------------------------- 
        block1_conv1 = Conv2D(64, 3, activation=activation_method, padding='same', name='block1_conv1', kernel_initializer='he_normal')(inputs)
        block1_conv2 = Conv2D(64, 3, activation=activation_method, padding='same', name='block1_conv2', kernel_initializer='he_normal')(block1_conv1)

        pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block1_conv2)

        block2_conv1 = Conv2D(128, 3, activation=activation_method, padding='same', name='block2_conv1', kernel_initializer='he_normal')(pool1)
        block2_conv2 = Conv2D(128, 3, activation=activation_method, padding='same', name='block2_conv2', kernel_initializer='he_normal')(block2_conv1)

        pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(block2_conv2)

        block3_conv1 = Conv2D(256, 3, activation=activation_method, padding='same', name='block3_conv1', kernel_initializer='he_normal')(pool2)
        block3_conv2 = Conv2D(256, 3, activation=activation_method, padding='same', name='block3_conv2', kernel_initializer='he_normal')(block3_conv1)
        block3_conv3 = Conv2D(256, 3, activation=activation_method, padding='same', name='block3_conv3', kernel_initializer='he_normal')(block3_conv2)

        pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(block3_conv3)

        block4_conv1 = Conv2D(512, 3, activation=activation_method, padding='same', name='block4_conv1', kernel_initializer='he_normal')(pool3)
        block4_conv2 = Conv2D(512, 3, activation=activation_method, padding='same', name='block4_conv2', kernel_initializer='he_normal')(block4_conv1)
        block4_conv3 = Conv2D(512, 3, activation=activation_method, padding='same', name='block4_conv3', kernel_initializer='he_normal')(block4_conv2)

        pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(block4_conv3)

        block5_conv1 = Conv2D(512, 3, activation=activation_method, padding='same', name='block5_conv1', kernel_initializer='he_normal')(pool4)
        block5_conv2 = Conv2D(512, 3, activation=activation_method, padding='same', name='block5_conv2', kernel_initializer='he_normal')(block5_conv1)
        block5_conv3 = Conv2D(512, 3, activation=activation_method, padding='same', name='block5_conv3', kernel_initializer='he_normal')(block5_conv2)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------

        
        up3_2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), name='up3_2', padding='same')(block4_conv3)
        merge3_2 = concatenate([up3_2, block3_conv3], name='merge3_2', axis=3)# axis=0~3 axis=3,前幾維不動第4維疊加
        #conv3_2 = standard_unit(merge3_2, '3_2')
        conv3_2 = standard_conv(merge3_2, stage='3_2', nb_filter = 256)                         
        conv3_2 = shortcut(input_tensor = conv3_2, shortcut_tensor = merge3_2, nb_filter = 256, stage='3_2')

        up2_2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), name='up2_2', padding='same')(block3_conv3)
        merge2_2 = concatenate([up2_2, block2_conv2], name='merge2_2', axis=3)
        #conv2_2 = standard_unit(merge2_2, '2_2')
        conv2_2 = standard_conv(merge2_2, stage='2_2', nb_filter = 128)        
        conv2_2 = shortcut(input_tensor = conv2_2, shortcut_tensor = merge2_2, nb_filter = 128, stage='2_2')
    
        up1_2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), name='up1_2', padding='same')(block2_conv2)
        merge1_2 = concatenate([up1_2, block1_conv2], name='merge1_2', axis=3)
        #conv1_2 = standard_unit(merge1_2, '1_2')
        conv1_2 = standard_conv(merge1_2, stage='1_2', nb_filter = 64)        
        conv1_2 = shortcut(input_tensor = conv1_2, shortcut_tensor = merge1_2, nb_filter = 64, stage='1_2')   
            
        up4_2 = Conv2DTranspose(512, (2, 2), strides=(2, 2), name='up4_2', padding='same')(block5_conv3)
        merge4_2 = concatenate([up4_2, block4_conv3], name='merge4_2', axis=3)
        #conv4_2 = standard_unit(merge4_2, '4_2')
        conv4_2 = standard_conv(merge4_2, stage='4_2', nb_filter = 512)        
        conv4_2 = shortcut(input_tensor = conv4_2, shortcut_tensor = merge4_2, nb_filter = 512, stage='4_2')   
        
        up3_3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), name='up3_3', padding='same')(conv4_2)
        merge3_3 = concatenate([up3_3, block3_conv3, conv3_2], name='merge3_3', axis=3)
        #conv3_3 = standard_unit(merge3_3, '3_3')
        conv3_3 = standard_conv(merge3_3, stage='3_3', nb_filter = 256)        
        conv3_3 = shortcut(input_tensor = conv3_3, shortcut_tensor = merge3_3, nb_filter = 256, stage='3_3')  
        
        up2_3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), name='up2_3', padding='same')(conv3_3)
        merge2_3 = concatenate([up2_3, block2_conv2, conv2_2], name='merge2_3', axis=3)
        #conv2_3 = standard_unit(merge2_3, '2_3')
        conv2_3 = standard_conv(merge2_3, stage='2_3', nb_filter = 128)  
        conv2_3 = shortcut(input_tensor = conv2_3, shortcut_tensor = merge2_3, nb_filter = 128, stage='2_3')  
        
        up1_3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), name='up1_3', padding='same')(conv2_3)
        merge1_3 = concatenate([up1_3, block1_conv2, conv1_2], name='merge1_3', axis=3)
        #conv1_3 = standard_unit(merge1_3, '1_3')
        conv1_3 = standard_conv(merge1_3, stage='1_3', nb_filter = 64)        
        conv1_3 = shortcut(input_tensor = conv1_3, shortcut_tensor = merge1_3, nb_filter = 64, stage='1_3')  
        
        #nestnet_output_1 = Conv2D(3, (1, 1), activation='softmax', name='output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=regularizers.l2(1e-4))(conv1_2)
        nestnet_output_2 = Conv2D(8, (1, 1), activation='softmax', name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=regularizers.l2(1e-4))(conv1_3)   #輸出為幾個class #類別
        print(nestnet_output_2.shape)

        
        if deep_supervision:
            model = Model(input=inputs, output=[nestnet_output_1,
                                                   nestnet_output_2])
        else:
            model = Model(input=inputs, output=[nestnet_output_2])
        
        for layer in model.layers[:19]:
            #print('layer:\n',layer)
            layer.trainable = True
        


        #model.load_weights('C:/Users/Admin/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)
        model.load_weights('./vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)
        #model.load_weights('D:/Admin/Desktop/unet-rgb/model/dataaug/vgg16_unet++3_ce_swish_10_13h5', by_name=True)
        #model.load_weights('./model/plants/vgg16_unet+_residual_7.hdf5')
        def categorical_focal_loss(gamma=5., alpha=1.):
            def categorical_focal_loss_fixed(y_true, y_pred):
                """
                :param y_true: A tensor of the same shape as `y_pred`
                :param y_pred: A tensor resulting from a softmax
                :return: Output tensor.
                """

                # Scale predictions so that the class probas of each sample sum to 1
                y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
                epsilon = K.epsilon()

                
                # Clip the prediction value to prevent NaN's and Inf's
                y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

                # Calculate Cross Entropy
                cross_entropy = -y_true * K.log(y_pred)

                # Calculate Focal Loss
                loss = alpha * y_true * K.pow(1 - y_pred, gamma) * cross_entropy

                # Sum the losses in mini_batch
                return K.sum(loss, axis=1)
            return categorical_focal_loss_fixed

        #model.compile(optimizer=Adam(lr=1e-4), loss=[categorical_focal_loss(alpha=1., gamma=5.)], metrics=['accuracy'])
        #model.compile(optimizer=Adam(lr=1e-5),loss = [focal_tversky_loss] , metrics=['accuracy', focal_tversky_loss])
        #model.load_weights('./model/dataaug/vgg16_unet++3_ce_swish_10.hdf5', by_name=True)
        model.compile(optimizer=Adam(lr=1e-4),loss = 'categorical_crossentropy' , metrics=['accuracy'])
        model.summary()
        return model
    

    def get_vgg19_unet(self):
        
        
        def swish(x):
            return (K.sigmoid(x) * x)
        activation_method = swish
        def standard_conv(input_tensor, stage, nb_filter, kernel_size=3):

            x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=activation_method, name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=regularizers.l2(1e-4))(input_tensor)
            #x = BatchNormalization()(x)
            #x = Dropout(dropout_rate, name='dp'+stage+'_1')(x)
            x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=activation_method, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
            #x = BatchNormalization()(x)
            #x = Dropout(dropout_rate, name='dp'+stage+'_2')(x)

            return x

        def shortcut(input_tensor, shortcut_tensor, nb_filter, stage, kernel_size=1):
            x = Conv2D(nb_filter, kernel_size = (1, 1), name = 'shortcut' + stage, padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4), strides=1)(shortcut_tensor)
            #x = BatchNormalization()(x)
            input_tensor = add([x, input_tensor])
            input_tensor = Activation(activation_method)(input_tensor)
            return input_tensor

        
        inputs = Input((self.img_rows, self.img_cols, 3))

        # If you want to specify input tensor shape, e.g. 256x256 with 3 channels:
        vgg_model = applications.VGG19(weights='imagenet', include_top=False, input_tensor=inputs)
        #print([layer.name for layer in vgg_model.layers])
        layers = dict([(layer.name, layer) for layer in vgg_model.layers])
        
        block5_conv4 = layers['block5_conv4'].output           #(None, 32, 32, 512)

        # Now getting bottom layers for multi-scale skip-layers
        block1_conv2 = layers['block1_conv2'].output       #(None, 512, 512, 64)
        block2_conv2 = layers['block2_conv2'].output      #(None, 256, 256, 128
        block3_conv4 = layers['block3_conv4'].output      #(None, 128, 128, 256
        block4_conv4 = layers['block4_conv4'].output      #(None, 64, 64, 512)
        
        up3_2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), name='up3_2', padding='same')(block4_conv4)
        merge3_2 = concatenate([up3_2, block3_conv4], name='merge3_2', axis=3)
        #conv3_2 = standard_unit(merge3_2, '3_2')
        conv3_2 = standard_conv(merge3_2, stage='3_2', nb_filter = 256)                         
        conv3_2 = shortcut(input_tensor = conv3_2, shortcut_tensor = merge3_2, nb_filter = 256, stage='3_2')

        up2_2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), name='up2_2', padding='same')(block3_conv4)
        merge2_2 = concatenate([up2_2, block2_conv2], name='merge2_2', axis=3)
        #conv2_2 = standard_unit(merge2_2, '2_2')
        conv2_2 = standard_conv(merge2_2, stage='2_2', nb_filter = 128)        
        conv2_2 = shortcut(input_tensor = conv2_2, shortcut_tensor = merge2_2, nb_filter = 128, stage='2_2')
    
        up1_2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), name='up1_2', padding='same')(block2_conv2)
        merge1_2 = concatenate([up1_2, block1_conv2], name='merge1_2', axis=3)
        #conv1_2 = standard_unit(merge1_2, '1_2')
        conv1_2 = standard_conv(merge1_2, stage='1_2', nb_filter = 64)        
        conv1_2 = shortcut(input_tensor = conv1_2, shortcut_tensor = merge1_2, nb_filter = 64, stage='1_2')   
            
        up4_2 = Conv2DTranspose(512, (2, 2), strides=(2, 2), name='up4_2', padding='same')(block5_conv4)
        merge4_2 = concatenate([up4_2, block4_conv4], name='merge4_2', axis=3)
        #conv4_2 = standard_unit(merge4_2, '4_2')
        conv4_2 = standard_conv(merge4_2, stage='4_2', nb_filter = 512)        
        conv4_2 = shortcut(input_tensor = conv4_2, shortcut_tensor = merge4_2, nb_filter = 512, stage='4_2')   
        
        up3_3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), name='up3_3', padding='same')(conv4_2)
        merge3_3 = concatenate([up3_3, block3_conv4, conv3_2], name='merge3_3', axis=3)
        #conv3_3 = standard_unit(merge3_3, '3_3')
        conv3_3 = standard_conv(merge3_3, stage='3_3', nb_filter = 256)        
        conv3_3 = shortcut(input_tensor = conv3_3, shortcut_tensor = merge3_3, nb_filter = 256, stage='3_3')  
        
        up2_3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), name='up2_3', padding='same')(conv3_3)
        merge2_3 = concatenate([up2_3, block2_conv2, conv2_2], name='merge2_3', axis=3)
        #conv2_3 = standard_unit(merge2_3, '2_3')
        conv2_3 = standard_conv(merge2_3, stage='2_3', nb_filter = 128)  
        conv2_3 = shortcut(input_tensor = conv2_3, shortcut_tensor = merge2_3, nb_filter = 128, stage='2_3')  
        
        up1_3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), name='up1_3', padding='same')(conv2_3)
        merge1_3 = concatenate([up1_3, block1_conv2, conv1_2], name='merge1_3', axis=3)
        #conv1_3 = standard_unit(merge1_3, '1_3')
        conv1_3 = standard_conv(merge1_3, stage='1_3', nb_filter = 64)        
        conv1_3 = shortcut(input_tensor = conv1_3, shortcut_tensor = merge1_3, nb_filter = 64, stage='1_3')  
        print(conv1_3.shape,end='\n')
        #nestnet_output_1 = Conv2D(3, (1, 1), activation='softmax', name='output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=regularizers.l2(1e-4))(conv1_2)
        output = Conv2D(3, (1, 1), activation='softmax', name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=regularizers.l2(1e-4))(conv1_3)   #輸出為幾個class
        
        model = Model(input=inputs, output=output)
        for layer in model.layers[:22]:
            print('layer:\n',layer)
            layer.trainable = True
        #model.load_weights('./model/unet_vgg19_2.hdf5')
        model.compile(optimizer=Adam(lr=1e-4),loss='categorical_crossentropy', metrics=['accuracy'])

        model.summary()
        return model
###############################

    def train(self):
        print("loading data")
        
        print("loading data done")
        model = self.get_vgg16_unet()
        #model = self.get_unet_resnet()
        #model = load_model('./model/unet_vgg19_2.hdf5')
        #model = self.get_vgg19_unet()
        print("got unet")
        #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        #print("plot_model_done")

        
        dataAugment = True
        #-------------------------------------------------------------------------------
        #-------------------------------------------------------------------------------
        #資料增強

        if(dataAugment):
            print('dataAugment')
            es = EarlyStopping(monitor='val_loss', patience=16, verbose=0, mode='auto')
            #model_checkpoint = ModelCheckpoint('./model/test.hdf5',monitor='loss', mode = 'min', verbose=1, save_best_only=True )
            model_checkpoint = ModelCheckpoint('./model/model_dataAugment.hdf5',monitor='loss', mode = 'min', verbose=1, save_best_only=True )
            reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience = 8, verbose=1, mode='auto')
            
            batch_size = 1
            #path = 'E:/畢業光碟製作/3. 論文實作目錄/輸入之影像視訊目錄/TRAINING_DATASETS'
            path= 'D:/unet(revised)/data/train_val(seperate)/'
            train_image_folder = '(reclassify)onlytrain_jpg'
            train_mask_folder = '(reclassify)onlytrain_label'

            val_image_folder = '(reclassify)val_jpg'
            val_mask_folder = '(reclassify)val_label'

            data_gen_args = dict(rotation_range = 10,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        fill_mode='nearest')
            
            traindata = trainGenerator(batch_size, path, train_image_folder, train_mask_folder, data_gen_args, save_to_dir = None)
            valdata = valGenerator(batch_size, path, val_image_folder, val_mask_folder, data_gen_args, save_to_dir = None)
   
            
            history = model.fit_generator(generator = traindata,
                                          #samples_per_epoch= 720,
                                          steps_per_epoch = 4096, #760
                                          epochs = 100,#100
                                          validation_data = valdata,
                                          validation_steps = 512,#
                                          callbacks=[es, model_checkpoint, reduce_lr_loss])
                 

        #-------------------------------------------------------------------------------
        #-------------------------------------------------------------------------------
        else:
            imgs_train, imgs_mask_train, imgs_test = self.load_data()
            model_checkpoint = ModelCheckpoint('./model/model_nondataAugment.hdf5', monitor='loss', verbose=1, save_best_only=True)
            reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, epsilon=1e-5, mode='auto')
            print('Fitting model...')
            history = model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=50, verbose=1,
                                validation_split=0.1, shuffle=True, callbacks=[model_checkpoint, reduce_lr_loss])
        
        
        #aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
        #        width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
        #        horizontal_flip=True, fill_mode="nearest")
        
        #model.fit_generator(aug.flow(trainX, imgs_mask_train, batch_size=1),steps_per_epoch=2000,epochs=5,callbacks=[model_checkpoint])
        #model.save_weights('./model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', save_best_only=True)
        print('training done')

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
         
        epochs = range(len(acc))
         
        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
         
        plt.figure()
         
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
         
        plt.show()

if __name__ == '__main__':
    myunet = myUnet()
    #model = myunet.get_vgg16_unet()
    # model.summary()
    # plot_model(model, to_file='model.png')
    myunet.train()
    #myunet.save_img()
