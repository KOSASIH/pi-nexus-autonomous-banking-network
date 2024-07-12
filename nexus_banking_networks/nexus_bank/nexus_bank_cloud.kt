import cloud.aws.s3.S3Client
import cloud.aws.s3.model.PutObjectRequest

class CloudStorage {
    private val s3Client = S3Client()

    fun uploadFile(bucketName: String, fileName: String, fileContent: ByteArray) {
        val request = PutObjectRequest(bucketName, fileName, fileContent)
        s3Client.putObject(request)
    }

    fun downloadFile(bucketName: String, fileName: String): ByteArray {
        val request = GetObjectRequest(bucketName, fileName)
        val response = s3Client.getObject(request)
        return response.objectContent.readBytes()
    }
}

fun main() {
    val cloudStorage = CloudStorage()
    val fileContent = "Hello, World!".toByteArray()
    cloudStorage.uploadFile("my-bucket", "hello.txt", fileContent)
    val downloadedFile = cloudStorage.downloadFile("my-bucket", "hello.txt")
    println(String(downloadedFile))
}
