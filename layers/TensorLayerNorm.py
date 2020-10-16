import tensorflow as tf

EPSILON = 0.00001

def tensor_layer_norm(x, state_name):
    x_shape = x.get_shape()
    //x为tensor，返回一个元组
    dims = x_shape.ndims
    //得到矩阵x_shape的维数
    params_shape = x_shape[-1:]//最后一维的大小
    if dims == 4:
        m, v = tf.nn.moments(x, [1,2,3], keep_dims=True)
        /求解x在维度1/2/3上的方差和均值，保持dims不变
    elif dims == 5:
        m, v = tf.nn.moments(x, [1,2,3,4], keep_dims=True)
        //求解x在维度1/2/3/4上的方差和均值，保持dims不变
    else:
        raise ValueError('input tensor for layer normalization must be rank 4 or 5.')
        //报错，维度必须为4或5
    //
    b = tf.get_variable(state_name+'b',initializer=tf.zeros(params_shape))
    s = tf.get_variable(state_name+'s',initializer=tf.ones(params_shape))
    //创建新的tensorflow变量,b的初始化使用全0的一维向量,s的初始化使用全1的一维向量
    x_tln = tf.nn.batch_normalization(x, m, v, b, s, EPSILON)
    //执行批归一化:x(input输入样本),m(mean样本均值),v(varinace样本方差),b(offset样本偏移)
    //s(scale缩放(默认是1)),EPSILON(variance_epsilon,为避免分母为零添加的一个极小值)
    //输出的计算公式：y = scale * (x - mean) / var + offset(y=s*(x-m)/v+b)
    return x_tln
