# Réseau Génératif Antagoniste (GAN) pour CelebA-HQ 

## Aperçu
Implémentation d'un GAN profond à convolution (DCGAN) pour générer des images de visages de haute qualité en utilisant le jeu de données CelebA-HQ.

## Caractéristiques
- Mécanisme d'auto-attention dans le Générateur et le Discriminateur
- Chargeur de données personnalisé pour CelebA-HQ
- Techniques d'augmentation d'images
- Support de l'entraînement CUDA et GPU MacOS

## Prérequis
- Python 3.8+
- PyTorch
- torchvision
- Pillow

## Commande d'entraînement
```bash
python train_dcgan1.py --dataset folder --dataroot ./CelebAMask-HQ/Subset --imageSize
 64 --batchSize 64 --niter 25 --lr 0.0002
```

### Arguments principaux
- `--dataset`: Dossier du jeu de données
- `--dataroot`: Chemin vers le jeu de données
- `--cuda`: Activer l'entraînement CUDA
- `--imageSize`: Taille d'image d'entrée (défaut : 64)
- `--batchSize`: Taille du lot d'entraînement (défaut : 64)
- `--niter`: Nombre d'époques d'entraînement (défaut : 25)

## Architecture du modèle
- **Générateur**: Convolution transposée avec auto-attention
- **Discriminateur**: Réseau convolutif avec sortie Sigmoid
- **Perte**: Entropie croisée binaire

## Processus d'entraînement
1. Charger et transformer les images
2. Entraîner le Discriminateur à distinguer les images réelles/fausses
3. Entraîner le Générateur à créer des images de plus en plus réalistes
4. Sauvegarder périodiquement les images générées et les points de contrôle du modèle

## Sortie
- Images générées sauvegardées dans le dossier de sortie
- Points de contrôle du modèle sauvegardés tous les 5 époques
