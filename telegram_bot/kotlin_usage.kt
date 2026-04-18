import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject

suspend fun submitKeystrokeCheckin(
    patientKeystrokeId: String,
    anomalyScore: Double,
    baseUrl: String = "https://wasurenai.uiutech.xyz",
    apiKey: String = "wasurenaiKEYSTROKEapiKEY123",
    mildThreshold: Double = 0.4,
    highThreshold: Double = 0.7,
    client: OkHttpClient = OkHttpClient()
): Result<String> = withContext(Dispatchers.IO) {
    try {
        require(mildThreshold < highThreshold) { "mildThreshold must be < highThreshold" }

        val status = when {
            anomalyScore < mildThreshold -> "normal"
            anomalyScore < highThreshold -> "mild_anomaly"
            else -> "high_anomaly"
        }

        val payload = JSONObject().apply {
            put("patient_keystroke_id", patientKeystrokeId)
            put("status", status)
            put("anomaly_score", anomalyScore)
        }

        val mediaType = "application/json; charset=utf-8".toMediaType()
        val body = payload.toString().toRequestBody(mediaType)

        val request = Request.Builder()
            .url("$baseUrl/api/keystroke-checkin")
            .addHeader("Content-Type", "application/json")
            .addHeader("X-API-Key", apiKey) // remove if server doesn't require it
            .post(body)
            .build()

        client.newCall(request).execute().use { response ->
            val responseBody = response.body?.string().orEmpty()
            if (response.isSuccessful) {
                Result.success(responseBody)
            } else {
                Result.failure(
                    Exception("HTTP ${response.code}: $responseBody")
                )
            }
        }
    } catch (e: Exception) {
        Result.failure(e)
    }
}