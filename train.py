import argparse
from trainer import Trainer
from model import MLPModel, EncoderModel
from dataloader import CustomDataset

train_dataset = CustomDataset()
valid_dataset = CustomDataset()

if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('--model_type', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    p.add_argument('--batch_size', default=128)
    p.add_argument('--n_epochs', default=256)

    p.add_argument('--savepath', default='./checkpoints/checkpoint')
    p.add_argument('--savepath_best', default='./checkpoints/best')
    p.add_argument('--checkpoint_epochs', default=20)

    p.add_argument('--load_path', default=None)

    p.add_argument('--hidden_size', default=512)
    p.add_argument('--nlayers', default=6)
    p.add_argument('--nheads', default=8)

    config = p.parse_args()

    if config.model_type == 'MLP':
        model = MLPModel(train_dataset.get_input_size,
                         config.hidden_size,
                         train_dataset.get_output_size,
                         config.nlayers)
    elif config.model_type == 'Encoder':
        model = EncoderModel(train_dataset.get_input_size,
                             config.hidden_size,
                             train_dataset.get_output_size,
                             config.nlayers,
                             config.nheads)
    else:
        raise Exception('model_type not defined')

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    train_dataloader = dataloader.create_dataloader(train_dataset,
                                                    batch_size=config.batch_size, shuffle=True)
    valid_dataloader = dataloader.create_dataloader(valid_dataset,
                                                    batch_size=config.batch_size, shuffle=True)

    trainer = Trainer(model, optimizer, criterion)
    trainer.train(train_dataloader, valid_dataloader, config)




