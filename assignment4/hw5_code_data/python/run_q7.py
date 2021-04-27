import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision
import torch.optim as optim
from scipy import io
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler


if __name__ == '__main__':
    
    training_for_question = 'q7_1_4'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training using", device)
        
    if training_for_question == 'q7_1_1':
        
        def get_random_batches(x,y,batch_size):
            batches = []
            inds = range(x.shape[0])
            while len(inds) > 0:
                rand_inds = np.random.randint(0, len(inds), batch_size)
                selected = [inds[i] for i in rand_inds]
                batch_x = [x[i] for i in selected]
                batch_y = [y[i] for i in selected]
                batches.append((np.array(batch_x), np.array(batch_y)))
                inds = list(set(inds) - set(selected))
            
            return batches

        def train_and_val_fn(net, epoch, train, batches, criterion, optimizer, device):
            
            transform = transforms.ToTensor()
            
            if train:
                net.train()
            else:
                net.eval()

            total_loss = 0.0
            correct = 0
            total = 0
            
            for xb, yb in tqdm(batches):
                #inputs, labels = data[0].to(device), data[1].to(device)
                inputs, labels = transform(np.float32(xb)), transform(np.float32(yb))
                inputs, labels = inputs.to(device), labels.to(device)
                inputs, labels = torch.squeeze(inputs), torch.squeeze(labels)
                labels = torch.argmax(labels, 1)
                #print('input', inputs.shape)
                #print('label', labels.shape)
                optimizer.zero_grad()
                outputs = net(inputs)
                #print('output', outputs.shape)
                if train:
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                else:
                    loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            average_loss = float(total_loss/len(batches))
            acc = 100 * correct / total
            return average_loss, acc

        batch_size = 32

        train_data = io.loadmat('../data/nist36_train.mat')
        valid_data = io.loadmat('../data/nist36_valid.mat')

        train_x, train_y = train_data['train_data'], train_data['train_labels']
        valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

        class TwoLayerNet(torch.nn.Module):
            def __init__(self, D_in, H, D_out):
                super(TwoLayerNet, self).__init__()
                self.linear1 = torch.nn.Linear(D_in, H)
                self.linear2 = torch.nn.Linear(H, D_out)

            def forward(self, x):
                h_relu = self.linear1(x).clamp(min=0)
                y_pred = self.linear2(h_relu)
                return y_pred

        net = TwoLayerNet(train_x.shape[1], 64, train_y.shape[1]).to(device)
        
        total_epoch = 200
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= total_epoch, verbose=True)
        
        SAVE_PATH = '../' + training_for_question + '.pth'

        best_acc = 0.0


        training_acc_list, validation_acc_list = [], []
        training_loss_list, validation_loss_list = [], []
        train_acc, val_acc = 0, 0
        for epoch in range(0, total_epoch):

            print("-"*50)
            print("Training Epoch", epoch)
            
            batches = get_random_batches(train_x, train_y, batch_size)
            batch_num = len(batches)
        
            average_train_loss, train_acc = train_and_val_fn(net = net, epoch = epoch, train = True, batches = batches, criterion=criterion, optimizer=optimizer, device = device)
            
            with torch.no_grad():
                average_val_loss, val_acc = train_and_val_fn(net = net, epoch = epoch, train = False, batches = batches, criterion=criterion, optimizer=optimizer, device = device)

            print("Average Training Loss :", average_train_loss, "Training acc :", train_acc, "%")
            print("Average Test Loss :", average_val_loss,  "Validation acc :", val_acc, "%")
            
            training_loss_list.append(average_train_loss)
            validation_loss_list.append(average_val_loss)

            training_acc_list.append(train_acc)
            validation_acc_list.append(val_acc)
            
            scheduler.step()

            if best_acc < val_acc:
                best_acc = val_acc
                torch.save(net.state_dict(), SAVE_PATH)
                print("saved weight to", SAVE_PATH)
            
            print("Best Acc :", best_acc)

        test_data = io.loadmat('../data/nist36_test.mat')
        test_x, test_y = test_data['test_data'], test_data['test_labels']
        batches = get_random_batches(test_x, test_y, batch_size)
        test_loss, test_acc = train_and_val_fn(net = net, epoch = 0, train = False, batches = batches, criterion=criterion, optimizer=optimizer, device = device)
        print("Test Loss :", test_loss)
        print("Train, Validation and Test acc :", train_acc, val_acc, test_acc)
        
        print("END")

        plt.figure()
        plt.plot(range(total_epoch), training_loss_list)
        plt.plot(range(total_epoch), validation_loss_list)
        plt.show()

        plt.figure()
        plt.plot(range(total_epoch), training_acc_list)
        plt.plot(range(total_epoch), validation_acc_list)
        plt.show()
                
                

    elif training_for_question == 'q7_1_2':
        def train_and_val_fn(net, epoch, train, loader, criterion, optimizer, device):
            
            t = tqdm(loader, file=sys.stdout)
            if train:
                t.set_description('Epoch %i %s' % (epoch, "Training"))
                net.train()
            else:
                t.set_description('Epoch %i %s' % (epoch, "Validation"))
                net.eval()            


            if train:
                net.train()
            else:
                net.eval()

            total_loss = 0.0
            correct = 0
            total = 0
            for i, data in enumerate(loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                inputs = inputs.repeat(1,3,1,1)
                #print('inputs', inputs.shape)
                #print('lables', labels.shape)
                optimizer.zero_grad()
                outputs = net(inputs)
                #print('outputs', outputs.shape)
                if train:
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                else:
                    loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            average_loss = float(total_loss/len(batches))
            acc = 100 * correct / total
            return average_loss, acc

        batch_size = 32

            
        net = torchvision.models.resnet18(pretrained = False, progress = True).to(device)
        net.fc = nn.Linear(512,10)
        total_epoch = 200
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= total_epoch, verbose=True)
        
        SAVE_PATH = '../' + training_for_question + '.pth'

        best_acc = 0.0

        train_transform = transforms.Compose([
            transforms.RandomSizedCrop(28),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485],
                                    std=[0.229]),
        ])

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485],
                                    std=[0.229]),
        ])

        valset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)

        validation_split = .2
        test_split = 0.8

        dataset_size = len(valset)
        indices = list(range(dataset_size))
        v_split = int(np.floor(validation_split * dataset_size))
        t_split = int(np.floor(test_split * dataset_size))
        val_indices = indices[0:v_split]
        test_indeices = indices[v_split:]
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indeices)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,sampler=val_sampler,
                                                    shuffle=False, num_workers=4)
        testloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,sampler=test_sampler,
                                                    shuffle=False, num_workers=4)
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                    download=True, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                    shuffle=True, num_workers=4)
        
        training_acc_list, validation_acc_list = [], []
        training_loss_list, validation_loss_list = [], []
        train_acc, val_acc = 0, 0
        for epoch in range(0, total_epoch):

            print("-"*50)
            print("Training Epoch", epoch)
            
        
            average_train_loss, train_acc = train_and_val_fn(net = net, epoch = epoch, train = True, loader = trainloader, criterion=criterion, optimizer=optimizer, device = device)
            
            with torch.no_grad():
                average_val_loss, val_acc = train_and_val_fn(net = net, epoch = epoch, train = False, loader = valloader, criterion=criterion, optimizer=optimizer, device = device)

            print("Average Training Loss :", average_train_loss, "Training acc :", train_acc, "%")
            print("Average Test Loss :", average_val_loss,  "Validation acc :", val_acc, "%")
            
            training_loss_list.append(average_train_loss)
            validation_loss_list.append(average_val_loss)

            training_acc_list.append(train_acc)
            validation_acc_list.append(val_acc)
            
            scheduler.step()

            if best_acc < val_acc:
                best_acc = val_acc
                torch.save(net.state_dict(), SAVE_PATH)
                print("saved weight to", SAVE_PATH)
            
            print("Best Acc :", best_acc)

        test_loss, test_acc = train_and_val_fn(net = net, epoch = 0, train = False, loader = testloader, criterion=criterion, optimizer=optimizer, device = device)
        print("Test Loss :", test_loss)
        print("Train, Validation and Test acc :", train_acc, val_acc, test_acc)
        
        print("END")

        plt.figure()
        plt.plot(range(total_epoch), training_loss_list)
        plt.plot(range(total_epoch), validation_loss_list)
        plt.show()

        plt.figure()
        plt.plot(range(total_epoch), training_acc_list)
        plt.plot(range(total_epoch), validation_acc_list)
        plt.show()
                
                
    elif training_for_question == 'q7_1_3':
        
        def get_random_batches(x,y,batch_size):
            batches = []
            inds = range(x.shape[0])
            while len(inds) > 0:
                rand_inds = np.random.randint(0, len(inds), batch_size)
                selected = [inds[i] for i in rand_inds]
                batch_x = [x[i] for i in selected]
                batch_y = [y[i] for i in selected]
                batches.append((np.array(batch_x), np.array(batch_y)))
                inds = list(set(inds) - set(selected))
            
            return batches

        def train_and_val_fn(net, epoch, train, batches, criterion, optimizer, device):
            
            transform = transforms.ToTensor()
            
            if train:
                net.train()
            else:
                net.eval()

            total_loss = 0.0
            correct = 0
            total = 0
            
            for xb, yb in tqdm(batches):
                #inputs, labels = data[0].to(device), data[1].to(device)
                inputs, labels = transform(np.float32(xb)), transform(np.float32(yb))
                inputs, labels = inputs.to(device), labels.to(device)
                inputs, labels = torch.squeeze(inputs), torch.squeeze(labels)
                batch_size = inputs.shape[0]
                inputs = torch.reshape(inputs, (batch_size, 1, 32, 32))
                inputs = inputs.repeat(1,3,1,1)
                labels = torch.argmax(labels, 1)
                #print('input', inputs.shape)
                #print('label', labels.shape)
                optimizer.zero_grad()
                outputs = net(inputs)
                #print('output', outputs.shape)
                if train:
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                else:
                    loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            average_loss = float(total_loss/len(batches))
            acc = 100 * correct / total
            return average_loss, acc

        batch_size = 32

        train_data = io.loadmat('../data/nist36_train.mat')
        valid_data = io.loadmat('../data/nist36_valid.mat')

        train_x, train_y = train_data['train_data'], train_data['train_labels']
        valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

        net = torchvision.models.resnet18(pretrained = False, progress = True).to(device)
        
        total_epoch = 200
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= total_epoch, verbose=True)
        
        SAVE_PATH = '../' + training_for_question + '.pth'

        best_acc = 0.0


        training_acc_list, validation_acc_list = [], []
        training_loss_list, validation_loss_list = [], []
        train_acc, val_acc = 0, 0
        for epoch in range(0, total_epoch):

            print("-"*50)
            print("Training Epoch", epoch)
            
            batches = get_random_batches(train_x, train_y, batch_size)
            batch_num = len(batches)
        
            average_train_loss, train_acc = train_and_val_fn(net = net, epoch = epoch, train = True, batches = batches, criterion=criterion, optimizer=optimizer, device = device)
            
            with torch.no_grad():
                average_val_loss, val_acc = train_and_val_fn(net = net, epoch = epoch, train = False, batches = batches, criterion=criterion, optimizer=optimizer, device = device)

            print("Average Training Loss :", average_train_loss, "Training acc :", train_acc, "%")
            print("Average Test Loss :", average_val_loss,  "Validation acc :", val_acc, "%")
            
            training_loss_list.append(average_train_loss)
            validation_loss_list.append(average_val_loss)

            training_acc_list.append(train_acc)
            validation_acc_list.append(val_acc)
            
            scheduler.step()

            if best_acc < val_acc:
                best_acc = val_acc
                torch.save(net.state_dict(), SAVE_PATH)
                print("saved weight to", SAVE_PATH)
            
            print("Best Acc :", best_acc)

        test_data = io.loadmat('../data/nist36_test.mat')
        test_x, test_y = test_data['test_data'], test_data['test_labels']
        batches = get_random_batches(test_x, test_y, batch_size)
        test_loss, test_acc = train_and_val_fn(net = net, epoch = 0, train = False, batches = batches, criterion=criterion, optimizer=optimizer, device = device)
        print("Test Loss :", test_loss)
        print("Train, Validation and Test acc :", train_acc, val_acc, test_acc)
        
        print("END")

        plt.figure()
        plt.plot(range(total_epoch), training_loss_list)
        plt.plot(range(total_epoch), validation_loss_list)
        plt.show()

        plt.figure()
        plt.plot(range(total_epoch), training_acc_list)
        plt.plot(range(total_epoch), validation_acc_list)
        plt.show()
                
                

    elif training_for_question == 'q7_1_4':
        def train_and_val_fn(net, epoch, train, loader, criterion, optimizer, device):
            
            t = tqdm(loader, file=sys.stdout)
            if train:
                t.set_description('Epoch %i %s' % (epoch, "Training"))
                net.train()
            else:
                t.set_description('Epoch %i %s' % (epoch, "Validation"))
                net.eval()            


            if train:
                net.train()
            else:
                net.eval()

            total_loss = 0.0
            correct = 0
            total = 0
            for i, data in enumerate(loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                inputs = inputs.repeat(1,3,1,1)
                #print('inputs', inputs.shape)
                #print('lables', labels.shape)
                optimizer.zero_grad()
                outputs = net(inputs)
                #print('outputs', outputs.shape)
                if train:
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                else:
                    loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            average_loss = float(total_loss/len(batches))
            acc = 100 * correct / total
            return average_loss, acc

        batch_size = 32

            
        net = torchvision.models.resnet18(pretrained = False, progress = True).to(device)
        net.fc = nn.Linear(512,47)

        total_epoch = 200
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= total_epoch, verbose=True)
        
        SAVE_PATH = '../' + training_for_question + '.pth'

        best_acc = 0.0

        train_transform = transforms.Compose([
            transforms.RandomSizedCrop(28),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485],
                                    std=[0.229]),
        ])

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485],
                                    std=[0.229]),
        ])

        valset = torchvision.datasets.EMNIST(root='./data', split = 'balanced', train=False,
                                        download=True, transform=transform)

        validation_split = .2
        test_split = 0.8

        dataset_size = len(valset)
        indices = list(range(dataset_size))
        v_split = int(np.floor(validation_split * dataset_size))
        t_split = int(np.floor(test_split * dataset_size))
        val_indices = indices[0:v_split]
        test_indeices = indices[v_split:]
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indeices)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,sampler=val_sampler,
                                                    shuffle=False, num_workers=4)
        testloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,sampler=test_sampler,
                                                    shuffle=False, num_workers=4)
        trainset = torchvision.datasets.EMNIST(root='./data', split = 'balanced', train=True,
                                                    download=True, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                    shuffle=True, num_workers=4)
        
        training_acc_list, validation_acc_list = [], []
        training_loss_list, validation_loss_list = [], []
        train_acc, val_acc = 0, 0
        for epoch in range(0, total_epoch):

            print("-"*50)
            print("Training Epoch", epoch)
            
        
            average_train_loss, train_acc = train_and_val_fn(net = net, epoch = epoch, train = True, loader = trainloader, criterion=criterion, optimizer=optimizer, device = device)
            
            with torch.no_grad():
                average_val_loss, val_acc = train_and_val_fn(net = net, epoch = epoch, train = False, loader = valloader, criterion=criterion, optimizer=optimizer, device = device)

            print("Average Training Loss :", average_train_loss, "Training acc :", train_acc, "%")
            print("Average Test Loss :", average_val_loss,  "Validation acc :", val_acc, "%")
            
            training_loss_list.append(average_train_loss)
            validation_loss_list.append(average_val_loss)

            training_acc_list.append(train_acc)
            validation_acc_list.append(val_acc)
            
            scheduler.step()

            if best_acc < val_acc:
                best_acc = val_acc
                torch.save(net.state_dict(), SAVE_PATH)
                print("saved weight to", SAVE_PATH)
            
            print("Best Acc :", best_acc)

        test_loss, test_acc = train_and_val_fn(net = net, epoch = 0, train = False, loader = testloader, criterion=criterion, optimizer=optimizer, device = device)
        print("Test Loss :", test_loss)
        print("Train, Validation and Test acc :", train_acc, val_acc, test_acc)
        
        print("END")

        plt.figure()
        plt.plot(range(total_epoch), training_loss_list)
        plt.plot(range(total_epoch), validation_loss_list)
        plt.show()

        plt.figure()
        plt.plot(range(total_epoch), training_acc_list)
        plt.plot(range(total_epoch), validation_acc_list)
        plt.show()


                
                        

        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches

        import skimage
        import skimage.measure
        import skimage.color
        import skimage.restoration
        import skimage.io
        import skimage.filters
        import skimage.morphology
        import skimage.segmentation
        import skimage.transform

        from nn import *
        from q4 import *
        # do not include any more libraries here!
        # no opencv, no sklearn, etc!
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=UserWarning)

        for img in os.listdir('../images'):
            im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
            bboxes, bw = findLetters(im1)

            plt.imshow(bw, cmap='gray')
            for bbox in bboxes:
                minr, minc, maxr, maxc = bbox
                rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                        fill=False, edgecolor='red', linewidth=2)
                plt.gca().add_patch(rect)
            plt.show()
            # find the rows using..RANSAC, counting, clustering, etc.
            heights = [bbox[2]-bbox[0] for bbox in bboxes]
            mean_height = sum(heights)/len(heights)
            # sort the bounding boxes with center y
            centers = [((bbox[2]+bbox[0])//2, (bbox[3]+bbox[1])//2, bbox[2]-bbox[0], bbox[3]-bbox[1]) for bbox in bboxes]
            centers = sorted(centers, key=lambda p: p[0])
            rows = []
            pre_h = centers[0][0]
            # cluster rows
            row = []
            for c in centers:
                if c[0] > pre_h + mean_height:
                    row = sorted(row, key=lambda p: p[1])
                    rows.append(row)
                    row = [c]
                    pre_h = c[0]
                else:
                    row.append(c)
            row = sorted(row, key=lambda p: p[1])
            rows.append(row)
            
            # crop the bounding boxes
            # note.. before you flatten, transpose the image (that's how the dataset is!)
            # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
            kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            data = []
            for row in rows:
                row_data = []
                for y, x, h, w in row:
                    # crop out the character
                    crop = bw[y-h//2:y+h//2, x-w//2:x+w//2]
                    # pad it to square
                    h_pad, w_pad = 0, 0
                    if h > w:
                        h_pad = h//20
                        w_pad = (h-w)//2+h_pad
                    elif h < w:
                        w_pad = w//20
                        h_pad = (w-h)//2+w_pad
                    crop = np.pad(crop, ((h_pad, h_pad), (w_pad, w_pad)), 'constant', constant_values=(1, 1))
                    # resize to 32*32
                    crop = skimage.transform.resize(crop, (32, 32))
                    crop = skimage.morphology.erosion(crop, kernel)
                    
                    crop = np.pad(crop, 4, 'constant', constant_values=(1, 1))
                    crop = skimage.transform.resize(crop, (32,32))
                    crop = crop * 3 - 2.0
                    crop[crop < 0.0] = 0.0

                    # plt.figure()
                    # plt.imshow(crop)
                    # plt.colorbar()
                    # plt.show()
                    crop = np.transpose(crop)
                    row_data.append(crop.flatten())
                data.append(np.array(row_data))
            
            # load the weights
            # run the crops through your neural network and print them out
            import pickle
            import string
            letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
            #params = pickle.load(open('q3_weights.pickle', 'rb'))

            ind2c = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
                    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
                    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3',
                    30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9', 36: 'a', 37:'b', 38:'c',
                    39: '4', 40: '5', 41: '6', 42: '7', 43: '8', 44: '9', 45: 'a', 46:'b'}
            for row_data in data:
                #print('row data', row_data.shape)
                batch_size = row_data.shape[0]
                row_data = torch.from_numpy(row_data).float()
                row_data = torch.reshape(row_data, (batch_size, 1, 32, 32))
                row_data = row_data.repeat(1,3,1,1).to(device)
                probs = net(row_data)
                #print(torch.argmax(probs, 1))
                probs = probs.detach().numpy()
                #h1 = forward(row_data, params, 'layer1')
                #probs = forward(h1, params, 'output', softmax)
                row_s = ''
                for i in range(probs.shape[0]):
                    ind = np.where(probs[i, :] == np.max(probs[i, :]))[0][0]
                    row_s += ind2c[ind]

                print(row_s)






