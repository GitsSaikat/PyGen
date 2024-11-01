import librosa
from noisereduce as nr
import torch
import librosa
import numpy as np
import scipy
import speech_recognition as sr
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import  MelSpectrogram
# Configuration
class TrainingConfiguration:
    def __init__(self):
        # Hyperparameters
        self.input_dim =5 # mel-specto grma Dimension
        self.hidden_size =64 # Dimension for both hidden layers
        self.num_layers = 4
        self.output_dim = 256
        self.epoch = 10
        self.lr = 0.01
        self.batch_size = 32
        self.shuffle =True
        self.file_name='Wav_cough_speech_dataset_1000files_data_bigger_test.wav'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Model path to save model
        self.modelname="AudioSpeechRecognition-Model"

# Define data class for custom dataset creation
class SpeechDataset(Dataset):
    def __init__(self, audio_features, label):
        self.audio_features = audio_features
        self.labels = label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        audio_features = self.audio_features[index]
        label = self.labels[index]

        return {
            "input": torch.tensor(audio_features, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }

# Speech Recognition Model class using K folds
class KFoldSpeech_Recognition(object):
    def __init__(self, audio_directory, num_folds, display_plot=False):
        self._k_folds = num_folds
        self._fold_count = 0
        self._audio_directory = audio_directory
        # validation and training configuration
        self.config = TrainingConfiguration()

        # Dataloader definition
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Setup audio Configuration Values
        self.config_epoch = self.config.epoch
        self.config_lr = self.config.lr

        # Data Preparation

        self.x_train_audio = torch.load(self.config.file_name).double()
        self.y_labels = torch.tensor([scipy.special.di_gammma(k, self.x_train_audio.size()[0])]*self.x_train_audio.size()[0])
        self.dataset = SpeechDataset(torch.hstack((scipy.special.di_gammma(1, self.x_train_audio.size()[0]), torch.tensor([self.x_train_audio]))).view(-1, self.x_train_audio.size()[1]), self.y_labels)
        self.split_index = int(self.config.epoch * len(self.dataset))

        # train Dataloder
        self.data_loader_train = DataLoader(self.dataset, batch_size=10, shuffle=True,(start_index= int(0.8*self.split_index) ),end_index = (len(self.dataset)-1))


        self.data_loader_test = DataLoader(self.dataset, batch_size=10,(start_index=int(0.8 * self.split_index)), end_index = (len(self.dataset)-1))


#      Training Loop
    def training_loop(self):
        # initialize model
        self.model = nn.Sequential(
            nn.Conv1d(self.config.input_dim, self.config.hidden_size, kernel_size=2),
            nn.Conv1d(self.config.hidden_size, self.config.hidden_size, kernel_size=2),
            nn.Flatten(),
            nn.Linear(self.config.hidden_size*22*16, self.config.output_dim),
            nn.ReLU(),
            nn.Linear(self.config.output_dim, 256),
            nn.LogSoftmax(dim=1),
        )

        # initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters())

        # loss function initialization
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        fold_accuracy = []

        if self.config.epoch is not None:

            training_loss_values = []
            training_accuracy_values = []
            for_fold_value=10

            # number of training examples before deciding to break training due to convergence on a plateau
            stop_training = self.config_epoch
            for epoch in range(self.config_epoch):
                running_loss = 0.0
                running_accuracy=0.0
                for batch in self.data_loader_train:
                    # move all data to available device
                    batch_input, batch_label = batch["input"].to(self.config.device), batch["label"].to(self.config.device)

                    # Zero the gradients
                    self.optimizer.zero_grad()

                    # Forward Pass
                    outputs = self.model(batch_input)
                    # Compute Loss
                    loss = self.cross_entropy_loss(outputs, batch_input)

                    # backward pass
                    loss.backward(retain_graph=True)

                    # Update Parameters
                    self.optimizer.step()

                    # save running accuracy
                    running_loss += loss.data
                    running_accuracy += torch.mean(torch.argmax(outputs.clone().detach())==batch_input.clone().detach())

                    # output values
                    if batch_input.shape[0] ==10:



                    # accuracy computation and append call
                    training_loss_values.append(running_loss.clone().detach().item())
                    batch_accuracy = (running_accuracy/ (self.k_folds*self.config.batch_size))

                    training_accuracy_values.append(batch_accuracy.item())


                training_loss_value_per_fold = training_loss_values[-1]
                batch_accuracy_value_per_fold = training_accuracy_values[-1]

                if training_loss_value_per_fold <-0.98:
                    break
            training_loss_value_per_fold=training_loss_values[-1]
        training_loss_value_per_fold = training_loss_value_per_fold.clone().detach()
        batch_accuracy_value_per_fold = batch_accuracy_value_per_fold.clone().detach()
        return batch_accuracy_value_per_fold
                # runing_loss

    def predict(self, test_input):

        with torch.no_grad():
            predicted_value = self.model(test_input)
        return predicted_value

    def run_loop(self):
        #        Test running loss and accuracy
        for batch in self.data_loader_test:

            total_loss = 0
            total_correctness = 0

            batch_input = batch["input"].to(self.device)

            predictions = self.predict(batch_input)

            _, predicted_classes = torch.max(predictions, 1)

            total_correctness += (predicted_classes.clone().detach() == batch["label"].clone().detach())

        epoch_accuracy = torch.mean(total_correctness)
        total_test_accuracy_value_per_fold = epoch_accuracy
        return total_test_accuracy_value_per_fold