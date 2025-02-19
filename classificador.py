import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Obtém o diretório base do script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "imagens")
TEST_IMAGE_DIR = os.path.join(BASE_DIR, "imagens_teste")
MODEL_PATH = os.path.join(BASE_DIR, "modelo.pth")

# Verifica se o diretório de imagens existe
if not os.path.exists(IMAGE_DIR):
    raise FileNotFoundError(f"O diretório '{IMAGE_DIR}' não foi encontrado.")

# Classes (nomes das pastas dentro de IMAGE_DIR)
CLASSES = sorted(os.listdir(IMAGE_DIR))  # Ordenar para garantir consistência
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}

# Dataset personalizado
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []

        for class_name in CLASSES:
            class_path = os.path.join(root_dir, class_name)
            if not os.path.exists(class_path):
                continue  # Ignorar pastas inexistentes

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if os.path.isfile(img_path):  # Apenas arquivos de imagem
                    self.images.append((img_path, CLASS_TO_IDX[class_name]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Erro ao carregar imagem: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertendo para RGB
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)  # Certifica que label é um tensor

# Transformações
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Criando o dataset e dataloader
dataset = ImageDataset(IMAGE_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Modelo simples de CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 64 * 64, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Criando modelo
num_classes = len(CLASSES)
model = SimpleCNN(num_classes)

# Definição do otimizador e da função de perda
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinamento manual
for images, labels in dataloader:
    plt.imshow(images[0].permute(1, 2, 0))  # Mostrar imagem
    plt.title(f"Classe: {CLASSES[labels[0].item()]}")
    plt.show()

    resposta = input("Digite a classe correta (ou pressione Enter para manter): ")
    if resposta and resposta in CLASSES:
        labels[0] = torch.tensor(CLASS_TO_IDX[resposta])  # Atualiza a label

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item():.4f}")

# Salvar modelo treinado
torch.save(model.state_dict(), MODEL_PATH)
print(f"Treinamento concluído! Modelo salvo em {MODEL_PATH}")

# Função para testar uma imagem
def predict_image(image_path, model, transform, class_names):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"A imagem '{image_path}' não foi encontrada.")

    model.eval()  # Modo de avaliação
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Erro ao carregar a imagem: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transform(image)  # Aplica as transformações
    image = image.unsqueeze(0)  # Adiciona uma dimensão extra para o batch

    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()  # Obtém a classe com maior probabilidade

    plt.imshow(image.squeeze().permute(1, 2, 0))
    plt.title(f"Classe prevista: {class_names[predicted_class]}")
    plt.show()

    return class_names[predicted_class]

# Carregar modelo treinado
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print(f"Modelo carregado de {MODEL_PATH}")

# Teste com uma imagem
test_image_path = os.path.join(TEST_IMAGE_DIR, "imagem_teste.jpg")
predicted_class = predict_image(test_image_path, model, transform, CLASSES)
print(f"A classe prevista é: {predicted_class}")
