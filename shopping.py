import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer (0)
        - Administrative_Duration, a floating point number (1)
        - Informational, an integer (2)
        - Informational_Duration, a floating point number (3)
        - ProductRelated, an integer (4)
        - ProductRelated_Duration, a floating point number (5)
        - BounceRates, a floating point number (6)
        - ExitRates, a floating point number (7)
        - PageValues, a floating point number (8)
        - SpecialDay, a floating point number (9)
        - Month, an index from 0 (January) to 11 (December) (10)
        - OperatingSystems, an integer (11)
        - Browser, an integer (12)
        - Region, an integer (13)
        - TrafficType, an integer (14)
        - VisitorType, an integer 0 (not returning) or 1 (returning) (15)
        - Weekend, an integer 0 (if false) or 1 (if true) (16)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []

    file = open(filename)
    reader = csv.reader(file)

    i = 0
    for row in reader:
        #print(row)
        if i != 0:
            data = row

            for i in [0, 2, 4, 11, 12, 13, 14]:
                data[i] = int(data[i])
            for i in [1, 3, 5, 6, 7, 8, 9]:
                data[i] = float(data[i])
            
            transalate = [{"Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5, "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11},
                          {"Returning_Visitor": 1, "New_Visitor": 0, "Other": 0},
                          {"TRUE": 1, "FALSE": 0}]
            data[10] = int(transalate[0][data[10]])

            data[15] = int(transalate[1][data[15]])

            data[16] = int(transalate[2][data[16]])

            evidence.append(data[:17])
            labels.append(data[17])
            #print(len(data))
        i += 1
    #print(len(labels))
    #print(len(evidence))
    #print(evidence[12329])
    file.close()
    return (evidence, labels)
    #raise NotImplementedError


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    #X_training = evidence
    #y_training = labels
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    truepositive = 0
    truenegative = 0
    falsepositive = 0
    falsenegative = 0

    total = len(predictions)

    for i in range(len(predictions)):
        #print()
        #print(predictions[i])
        #print(labels[i])
        if (predictions[i] == "TRUE") and (labels[i] == "TRUE"):
            truepositive += 1
            #print("TP")
        if (predictions[i] == "FALSE") and (labels[i] == "TRUE"):
            falsenegative += 1
            #print("FN")
        if (predictions[i] == "TRUE") and (labels[i] == "FALSE"):
            falsepositive += 1
            #print("FP")
        if (predictions[i] == "FALSE") and (labels[i] == "FALSE"):
            truenegative += 1
            #print("TN")

    sensitivity = truepositive / (truepositive + falsenegative)
    specificity = truenegative / (falsepositive + truenegative)

    return (sensitivity, specificity)
        




if __name__ == "__main__":
    main()
