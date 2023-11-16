
def caculate_model_size(model):
    param_size = 0
    for _, param in model.named_parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    print('param size: {:.3f}MB'.format(param_size / 1024 ** 2))
    print('buffer size: {:.3f}MB'.format(buffer_size / 1024 ** 2))
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb))