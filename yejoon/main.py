from run import RunFreeze, RunLora, RunAdapter

# config = dict(
#     num_freeze = 9,
#     lr = 1e-5,
#     batch_size = 8,
#     weight_decay = 5e-6,
#     hgf_checkpoint='bert-base-multilingual-cased',
#     num_epochs = 8,
#     n_log = 2000)

# run = RunFreeze(config=config, project='Parameter-Efficient-Tuning')

config = dict(
    lora_r = 16,
    lora_module_names = ['query', 'value'],
    lr = 5e-4,
    batch_size = 8,
    weight_decay = 5e-6,
    hgf_checkpoint='bert-base-multilingual-cased',
    num_epochs = 8,
    n_log = 2000)

run = RunLora(config=config, project='Parameter-Efficient-Tuning', name='lora-16')

# config = dict(
#     adapter_dim = 1,
#     lr = 1e-4,
#     batch_size = 8,
#     weight_decay = 5e-6,
#     hgf_checkpoint = 'bert-base-multilingual-cased',
#     num_epochs = 8,
#     n_log = 2000
# )

# run = RunAdapter(config=config, project='Parameter-Efficient-Tuning', name='adapter-1')