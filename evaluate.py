# Produces a classification report of the predictions
import argparse
import os
from pickle import load
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def get_args():
    parser = argparse.ArgumentParser(description="Evaluates the performance of given predictions. Results are printed to stdout")

    parser.add_argument("predictions", help="File produced by predict.py")
    parser.add_argument("--plot", help="If defined, the confusion matrix is saved with the given name")

    return parser.parse_args()

def plot_confusion(conf_matrix, file_name: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
    plt.xlabel('Prédiction', fontsize=18)
    plt.ylabel('Vérité', fontsize=18)
    plt.title('Matrice de confusion', fontsize=18)
    
    # Save plot
    save_folder, _ = os.path.split(file_name)
    os.makedirs(save_folder, exist_ok=True)
    
    plt.savefig(file_name)

def main():
    args = get_args()

    # Open the predictions
    file = load(open(args.predictions, 'rb'))

    truth, predicted = file
    print(classification_report(truth, predicted))
    conf_matrix = confusion_matrix(truth, predicted)
    print(conf_matrix)
    if args.plot is not None:
        plot_confusion(conf_matrix, args.plot)

if __name__ == "__main__":
    main()