import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import models
from PIL import Image
import matplotlib.pyplot as plt



def is_football(x, model, preprocess,threshold=0.5):
    x = preprocess(x)
    print(type(x))
    print(x.shape)

    if model(x) > threshold:
        return True
    return False

def is_event(model, preprocess, x):
    x = preprocess(x)
    if model(x) == 0:
        return False
    return True

def event_detection(model, preprocess, x):
    x = preprocess(x)
    pred = model(x)
    if pred == 0:
        return False
    return pred

def card(model, preprocess, x):
    x = preprocess(x)
    return model(x)

def main():
    """
    First we check if the image is football or not -> is_football() by using the VAE model
    Then we check if there is an event -> event_detection() by using the event detection model (CNN)
    If the event is a card, we check which card it is -> card() by using the card model (B-CNN or other fine-grained models)
    """
    # Load models
    vqvae_model = models.vqvae
    event_detection_model = models.nfnet
    event_classification_model = models.vit
    card_model = models.FG

    vqvae_model.eval()
    event_detection_model.eval()
    event_classification_model.eval()
    card_model.eval()


    # Load images
    image = Image.open('./trial/Corner/Corner__1__55.jpg')
    plt.imshow(image)
    plt.show()

    if is_football(image, models.vqvae_transform ,vqvae_model):
        if is_event(event_detection_model, models.nfnet_transform, image):
            event = event_detection(event_classification_model, models.processor, image)
            if event == 1:
                card = card(card_model, models.FG_transform, image)
                print(card)
            else:
                print(event)
        else:
            print('No event')
    else:
        print('Not football')

    return 0

if __name__ == "__main__":
    main()
    