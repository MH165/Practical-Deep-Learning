from fastai.vision.all import *
from fastcore.all import *

# ─────────────────────────────────────────────
# Download mnis dataset
path = untar_data(URLs.MNIST_SAMPLE)
Path.BASE_PATH = path

threes = (path/'train'/'3').ls()
sevens = (path/'train'/'7').ls()

three_tensors = [torch.tensor(np.array(Image.open(p))) for p in threes]
seven_tensors = [torch.tensor(np.array(Image.open(p))) for p in sevens]
# print(f"Number of '3' images: {len(three_tensors)}")
# print(f"Number of '7' images: {len(seven_tensors)}")

# show_image(three_tensors[0])
# plt.show()
# stack the tensores into a single tensor and normalize the values to be between 0 and 1

three_stack = torch.stack(three_tensors).float()/255
seven_stack= torch.stack(seven_tensors).float()/255

# print(f"Stacked '3' tensor shape: {three_stack.shape}")
# print(f"Stacked '7' tensor shape: {seven_stack.shape}")

mean3 = three_stack.mean(0)
mean7 = seven_stack.mean(0)
# show_image(mean3)
# plt.show()
# show_image(mean7)
# plt.show()

# calculate the difference between the two mean images
sampleImage = three_stack[0]
# difference = (sampleImage - mean3).abs().mean()
# print(f"Average absolute difference between sample image and mean '3': {difference.item():.4f}")

# difference7 = (sampleImage - mean7).abs().mean()
# print(f"Average absolute difference between sample image and mean '7': {difference7.item():.4f}")

# calculate the L1 loss between the sample image and the mean images
# diff3 = F.l1_loss(sampleImage, mean3)
# diff7 = F.l1_loss(sampleImage, mean7)
# print(f"L1 loss between sample image and mean '3': {diff3.item():.4f}")
# print(f"L1 loss between sample image and mean '7': {diff7.item():.4f}")

# # calculate the L2 loss between the sample image and the mean images
# diff3_l2 = F.mse_loss(sampleImage, mean3)
# diff7_l2 = F.mse_loss(sampleImage, mean7)
# print(f"L2 loss between sample image and mean '3': {diff3_l2.item():.4f}")
# print(f"L2 loss between sample image and mean '7': {diff7_l2.item():.4f}")

# calculate the cosine similarity between the sample image and the mean images
# cosine_sim3 = F.cosine_similarity(sampleImage.flatten(), mean3.flatten(), dim=0)
# cosine_sim7 = F.cosine_similarity(sampleImage.flatten(), mean7.flatten(), dim=0)
# print(f"Cosine similarity between sample image and mean '3': {cosine_sim3.item():.4f}")
# print(f"Cosine similarity between sample image and mean '7': {cosine_sim7.item():.4f}")

# Using validating set

valid3 = (path/'valid'/'3').ls()
valid7 = (path/'valid'/'7').ls()

valid3_tensors = [torch.tensor(np.array(Image.open(p))) for p in valid3]
valid7_tensors = [torch.tensor(np.array(Image.open(p))) for p in valid7]

valid3_stack = torch.stack(valid3_tensors).float()/255
valid7_stack = torch.stack(valid7_tensors).float()/255

train_images = torch.cat([three_stack, seven_stack]).view(-1, 28*28)
train_labels = torch.tensor([1]*len(threes) + [0]*len(sevens)).unsqueeze(1)

training_dset = list(zip(train_images, train_labels))

valid_images = torch.cat([valid3_stack, valid7_stack]).view(-1, 28*28)
valid_labels = torch.tensor([1] * len(valid3) + [0] * len(valid7)).unsqueeze(1)
validation_dset = list(zip(valid_images, valid_labels))

def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()

train_dl = DataLoader(training_dset, batch_size=256)
valid_dl = DataLoader(validation_dset, batch_size=256)

def cal_grad(xb, yb, model):
    pred = model(xb)
    loss = mnist_loss(pred, yb)
    loss.backward()

def batch_accuracy(xb, yb):
    pred = xb.sigmoid()
    corrects = (pred > 0.5) == yb
    return corrects.float().mean()

def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb, yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)

linear_model = nn.Linear(28*28, 1)

class BasicOptim:
    def __init__(self, params, lr):
        self.params, self.lr = list(params), lr
    def step(self, *args, **kwargs):
        for p in self.params:
            p.data -= self.lr * p.grad.data
    def zero_grad(self, *args, **kwargs):
        for p in self.params:
            p.grad = None
    
opt = BasicOptim(linear_model.parameters(), lr=1)
def train_epoch(model):
    for xb, yb in train_dl:
        cal_grad(xb, yb, model)
        opt.step()
        opt.zero_grad()

def train_model(model, epochs):
    for i in range(epochs):
        train_epoch(model)
        acc = validate_epoch(model)
        print(f"Epoch {i+1}: Validation Accuracy: {acc:.4f}")

train_model(linear_model, 20)
