import torch


class Trainer:
    def __init__(self, model, optimizer, criterion, config):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config

        super().__init__()

    def save(self, epoch, save_path):
        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch}, save_path + '.pt')

    def __train(self, train_dataloader):
        self.model.train()

        total_loss = 0

        for train_x, train_y in train_dataloader:

            train_x, train_y = train_x.to(self.config.device), train_y.to(self.config.device)

            label_pred = self.model(train_x)

            mask = torch.where(train_y != -1, 1.0, 0.0).to(self.config.device)
            loss = self.criterion(label_pred, train_y)
            loss = (loss * mask).sum() / mask.sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += (loss.item() / len(train_x))

        return total_loss

    def __validate(self, valid_dataloader):
        self.model.eval()

        total_loss = 0

        with torch.no_grad():
            for valid_x, valid_y in valid_dataloader:

                valid_x, valid_y = valid_x.to(self.config.device), valid_y.to(self.config.device)

                label_pred = self.model(valid_x)

                mask = torch.where(valid_y != -1, 1.0, 0.0).to(self.config.device)
                loss = self.criterion(label_pred, valid_y)
                loss = (loss * mask).sum() / mask.sum()

                total_loss += (loss.item() / len(valid_x))

        return total_loss

    def train(self, train_dataloader, valid_dataloader):
        best_loss = float('inf')
        checkpoints = range(0, self.config.n_epochs, self.config.checkpoint_epochs)
        epoch = 0

        if self.config.load_path is not None:
            checkpoint = torch.load(self.config.load_path, map_location=self.config.device)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch']

        while epoch < self.config.n_epochs:
            epoch += 1

            train_loss = self.__train(train_dataloader)
            valid_loss = self.__validate(valid_dataloader)

            if valid_loss < best_loss:
                best_loss = valid_loss
                self.save(epoch, self.config.savepath_best)

            if epoch in checkpoints:
                self.save(epoch, self.config.savepath + str(epoch))

            print('Epoch[%d/%d] train_loss: %.5f, valid_loss: %.5f' %
                  (epoch, self.config.n_epochs, train_loss, valid_loss))

