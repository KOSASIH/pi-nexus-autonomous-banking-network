import sys
from quantum_algorithms.grovers_algorithm import grovers_algorithm
from quantum_algorithms.shors_algorithm import shors_algorithm
from quantum_algorithms.quantum_teleportation import quantum_teleportation
from quantum_cryptography.quantum_key_distribution import quantum_key_distribution
from quantum_cryptography.quantum_encryption import quantum_encryption
from quantum_machine_learning.quantum_svm import quantum_svm
from quantum_machine_learning.quantum_neural_network import quantum_neural_network

def main():
    print("Welcome to the Quantum Integration Project!")
    print("Please choose an option:")
    print("1. Grover's Algorithm")
    print("2. Shor's Algorithm")
    print("3. Quantum Teleportation")
    print("4. Quantum Key Distribution")
    print("5. Quantum Encryption")
    print("6. Quantum Support Vector Machine")
    print("7. Quantum Neural Network")
    print("0. Exit")

    while True:
        choice = input("Enter your choice (0-7): ")

        if choice == '1':
            n = int(input("Enter the number of qubits: "))
            target = int(input("Enter the target index (0 to n-1): "))
            counts = grovers_algorithm(n, target)
            print("Grover's Algorithm Results:", counts)

        elif choice == '2':
            N = int(input("Enter the number to factor (N): "))
            counts = shors_algorithm(N)
            print("Shor's Algorithm Results:", counts)

        elif choice == '3':
            counts = quantum_teleportation()
            print("Quantum Teleportation Results:", counts)

        elif choice == '4':
            num_bits = int(input("Enter the number of bits for QKD: "))
            bits, bases, counts = quantum_key_distribution(num_bits)
            print("Alice's Bits:", bits)
            print("Alice's Bases:", bases)
            print("Measurement Results:", counts)

        elif choice == '5':
            message = input("Enter the message to encrypt: ")
            encrypted_message, key = quantum_encryption(message)
            print("Original Message:", message)
            print("Encrypted Message:", encrypted_message)
            print("Key Used:", key)

        elif choice == '6':
            # Generate synthetic data for classification
            from sklearn.datasets import make_classification
            from sklearn.model_selection import train_test_split

            X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            predictions = quantum_svm(X_train, y_train)
            accuracy = accuracy_score(y_train, predictions)
            print("Training Accuracy:", accuracy)

        elif choice == '7':
            # Generate synthetic data for classification
            from sklearn.datasets import make_moons
            from sklearn.model_selection import train_test_split

            X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            predictions = quantum_neural_network(X_train, y_train, X_test)
            accuracy = accuracy_score(y_test, predictions)
            print("Test Accuracy:", accuracy)

        elif choice == '0':
            print("Exiting the program.")
            sys.exit()

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
