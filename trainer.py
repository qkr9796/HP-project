import torch


class Trainer:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        super().__init__()

    def save(self, epoch, save_path):
        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch}, save_path + '.pt')

    def __train(self, train_dataloader):
        self.model.train()

        total_loss = 0

        for train_x, train_y in train_dataloader:

            label_pred = self.model(train_x)

            loss = self.criterion(label_pred, train_y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += ( loss.item() / len(train_x) )

        return total_loss

    def __validate(self, valid_dataloader):
        self.model.eval()

        total_loss = 0

        with torch.no_grad():
            for valid_x, valid_y in train_datalodaer:

                label_pred = self.model(valid_x)

                loss = self.criterion(label_pred, valid_y)

                total_loss += (loss.item() / len(valid_x))

        return total_loss

    def train(self, train_dataloader, valid_dataloader, config):
        best_loss = float('inf')
        checkpoints = range(0, config.n_epochs, config.checkpoint_epochs)
        epoch = 0

        if config.load_path is not None:
            checkpoint = torch.load(load_path)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(check['optimizer'])
            epoch = checkpoint['epoch']

        while epoch < config.n_epochs:
            epoch += 1

            train_loss = self.__train(train_dataloader)
            valid_loss = self.__validate(valid_dataloader)

            if valid_loss < best_loss:
                best_loss = valid_loss
                self.save(epoch, config.savepath_best)

            if epoch in checkpoints:
                self.save(epoch, config.savepath + str(epoch))

            print('Epoch[%d/%d] train_loss: %.5f, valid_loss: %.5f' %
                  (epoch, config.n_epochs, train_loss, valid_loss))

