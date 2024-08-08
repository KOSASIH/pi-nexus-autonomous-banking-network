import pyseal

class HomomorphicEncryption:
    def __init__(self, public_key, private_key):
        self.public_key = public_key
        self.private_key = private_key
        self.context = pyseal.SEALContext.Create()
        self.key_context = pyseal.KeyContext(self.context)
        self.encryptor = pyseal.Encryptor(self.context, self.public_key)
        self.decryptor = pyseal.Decryptor(self.context, self.private_key)
        self.evaluator = pyseal.Evaluator(self.context)

    def encrypt(self, plaintext):
        ciphertext = pyseal.Ciphertext()
        self.encryptor.encrypt(plaintext, ciphertext)
        return ciphertext

    def decrypt(self, ciphertext):
        plaintext = pyseal.Plaintext()
        self.decryptor.decrypt(ciphertext, plaintext)
        return plaintext

    def add(self, ciphertext1, ciphertext2):
        result = pyseal.Ciphertext()
        self.evaluator.add(ciphertext1, ciphertext2, result)
        return result

    def multiply(self, ciphertext1, ciphertext2):
        result = pyseal.Ciphertext()
        self.evaluator.multiply(ciphertext1, ciphertext2, result)
        return result
