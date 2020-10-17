__author__ = 'yunbo'

import tensorflow as tf
from layers.TensorLayerNorm import tensor_layer_norm

class CausalLSTMCell():
    def __init__(self, layer_name, filter_size, num_hidden_in, num_hidden_out,
                 seq_shape, forget_bias=1.0, tln=False, initializer=0.001):
        """Initialize the Causal LSTM cell.
        Args:
            layer_name: layer names for different lstm layers.
            filter_size: int tuple thats the height and width of the filter.//滤波器的长宽值
            num_hidden_in: number of units for input tensor.//输入张量的单元数
            num_hidden_out: number of units for output tensor.//输出张量的单元数
            seq_shape: shape of a sequence.//序列的形状
            forget_bias: float, The bias added to forget gates.//遗忘门上添加偏置
            tln: whether to apply tensor layer normalization //是否使用张量层归一化
        """
        self.layer_name = layer_name
        self.filter_size = filter_size
        self.num_hidden_in = num_hidden_in
        self.num_hidden = num_hidden_out
        self.batch = seq_shape[0]//batch即序列的第一维
        self.height = seq_shape[2]
        self.width = seq_shape[3]
        self.layer_norm = tln
        self._forget_bias = forget_bias
        self.initializer = tf.random_uniform_initializer(-initializer,initializer)//初始化的程序

        //定义一个初始状态的四维矩阵：[seq_shape[0],seq_shape[2],seq_shape[3],num_hidden_out,]
    def init_state(self):
        return tf.zeros([self.batch, self.height, self.width, self.num_hidden],
                        dtype=tf.float32)
    

    //调用x,h,c,m变量
    def __call__(self, x, h, c, m):
        //如果h不为None，可以定义为全为0的四维向量h
        //h：[seq_shape[0],seq_shape[2],seq_shape[3],num_hidden_out]
        if h is None:
            h = tf.zeros([self.batch, self.height, self.width,
                          self.num_hidden],
                         dtype=tf.float32)
        if c is None:
            c = tf.zeros([self.batch, self.height, self.width,
                          self.num_hidden],
                         dtype=tf.float32)
        //m:[seq_shape[0],seq_shape[2],seq_shape[3],num_hidden_in]
        if m is None:
            m = tf.zeros([self.batch, self.height, self.width,
                          self.num_hidden_in],
                         dtype=tf.float32)

        //layer_name,方便管理参数，更好的封装
        with tf.variable_scope(self.layer_name):
            //inputs = h(四维向量);filters输出空间的维数：num_hidden_out*4;
            //kernel_size卷积窗的高和宽：filter_size;strides:卷积的横纵向的步长（一个整数：横纵相等）
            //padding:'same'表示不够卷积大小的块就补'0';kernel_initializer:卷积核的初始化，使用上述的初始化函数
            //name:'temporal_state_transition',时间状态转换
            //创建了一个卷积核，将输入进行卷积来输出一个 tensor
            h_cc = tf.layers.conv2d(
                h, self.num_hidden*4,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer,
                name='temporal_state_transition')
            c_cc = tf.layers.conv2d(
                c, self.num_hidden*3,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer,
                name='temporal_memory_transition')
            //name:时间记忆转换
            m_cc = tf.layers.conv2d(
                m, self.num_hidden*3,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer,
                name='spatial_memory_transition')
            //name:空间状态转换
            if self.layer_norm:
                //调用TensorLayerNorm中的函数
                //批归一化
                h_cc = tensor_layer_norm(h_cc, 'h2c')
                c_cc = tensor_layer_norm(c_cc, 'c2c')
                m_cc = tensor_layer_norm(m_cc, 'm2m')
            //tf.split,把一个张量切分成几份，h_cc准备切分的张量,切成4份，在第三个维度上切
            tf.split(value,num_or_size_splits,axis=0,num=None,name='split')
            //value:准备切分的张量,num_or_size_splits:切成几份,axis:在第几个维度上切
            i_h, g_h, f_h, o_h = tf.split(h_cc, 4, 3)
            i_c, g_c, f_c = tf.split(c_cc, 3, 3)
            i_m, f_m, m_m = tf.split(m_cc, 3, 3)

            if x is None:
                //计算x元素的Sigmoid 
                i = tf.sigmoid(i_h + i_c)
                f = tf.sigmoid(f_h + f_c + self._forget_bias)
                g = tf.tanh(g_h + g_c)
            else:
                //name:'输入状态'
                x_cc = tf.layers.conv2d(
                    x, self.num_hidden*7,
                    self.filter_size, 1, padding='same',
                    kernel_initializer=self.initializer,
                    name='input_to_state')
                if self.layer_norm:
                    x_cc = tensor_layer_norm(x_cc, 'x2c')

                i_x, g_x, f_x, o_x, i_x_, g_x_, f_x_ = tf.split(x_cc, 7, 3)
                //计算元素的Sigmoid
                i = tf.sigmoid(i_x + i_h + i_c)
                f = tf.sigmoid(f_x + f_h + f_c + self._forget_bias)
                g = tf.tanh(g_x + g_h + g_c)
            //更新c的状态
            c_new = f * c + i * g
            //c到m的状态转变
            c2m = tf.layers.conv2d(
                c_new, self.num_hidden*4,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer,
                name='c2m')
            if self.layer_norm:
                c2m = tensor_layer_norm(c2m, 'c2m')

            i_c, g_c, f_c, o_c = tf.split(c2m, 4, 3)
            //Sigmoid和tanh
            if x is None:
                ii = tf.sigmoid(i_c + i_m)
                ff = tf.sigmoid(f_c + f_m + self._forget_bias)
                gg = tf.tanh(g_c)
            else:
                ii = tf.sigmoid(i_c + i_x_ + i_m)
                ff = tf.sigmoid(f_c + f_x_ + f_m + self._forget_bias)
                gg = tf.tanh(g_c + g_x_)
            //m的状态更新
            m_new = ff * tf.tanh(m_m) + ii * gg
            //o到m的状态更新
            o_m = tf.layers.conv2d(
                m_new, self.num_hidden,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer,
                name='m_to_o')
            if self.layer_norm:
                o_m = tensor_layer_norm(o_m, 'm2o')
            //根据x的状态选择不同的o
            if x is None:
                o = tf.tanh(o_h + o_c + o_m)
            else:
                o = tf.tanh(o_x + o_h + o_c + o_m)
            //将新的c和m的在最后一维拼接起来
            cell = tf.concat([c_new, m_new],-1)
            cell = tf.layers.conv2d(cell, self.num_hidden, 1, 1,
                                    padding='same', name='memory_reduce')
            //h状态的更新
            h_new = o * tf.tanh(cell)
            //返回，更新后的h,c,m
            return h_new, c_new, m_new


