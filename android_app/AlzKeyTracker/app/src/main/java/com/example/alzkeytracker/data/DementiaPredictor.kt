package com.example.alzkeytracker.data

import com.example.alzkeytracker.database.KeystrokeEntity
import kotlin.math.sqrt

/**
 * Nearest-centroid classifier for cognitive decline detection.
 *
 * THREE CATEGORIES
 * ─────────────────────────────────────────────────────────────────────────
 * Healthy          — typing patterns within normal ageing range
 * Pre-dementia     — patterns consistent with Mild Cognitive Impairment (MCI);
 *                    this is the critical early-detection target
 * Moderate Decline — patterns consistent with moderate cognitive decline /
 *                    Alzheimer's disease stage
 *
 * HOW THE CLASSIFIER WORKS
 * ─────────────────────────────────────────────────────────────────────────
 * Six features are extracted from a keystroke session:
 *   0. Mean hold duration (ms)          — motor speed
 *   1. CV of hold duration              — motor consistency
 *   2. Mean inter-key interval (ms)     — processing speed
 *   3. CV of inter-key interval         — rhythm regularity
 *   4. Error rate (backspace fraction)  — error correction load
 *   5. Long-pause rate (IKI > 1500 ms)  — word-finding hesitations
 *
 * The feature vector is normalised and compared to three centroids.
 * Centroid values are derived analytically from clinical-literature parameter
 * ranges (Arroyo-Gallego et al. 2017, Garn et al. 2019) and validated against
 * synthetic profiles generated during development.
 *
 * This is NOT a medical diagnostic tool. Results are indicators for research.
 */
object DementiaPredictor {

    // ── Centroids ──────────────────────────────────────────────────────────

    data class Centroid(
        val label: String,
        val displayName: String,
        val features: DoubleArray
    )

    private val CENTROIDS = listOf(
        Centroid(
            label = "healthy",
            displayName = "Healthy",
            //          hold  cvHold   IKI   cvIKI  errRate  longPause
            features = doubleArrayOf(85.0, 0.212, 250.0, 0.220, 0.04, 0.02)
        ),
        Centroid(
            label = "early_mci",
            displayName = "Pre-dementia (Early MCI)",
            features = doubleArrayOf(115.0, 0.304, 420.0, 0.286, 0.09, 0.08)
        ),
        Centroid(
            label = "moderate_ad",
            displayName = "Moderate Cognitive Decline",
            features = doubleArrayOf(175.0, 0.371, 780.0, 0.359, 0.18, 0.18)
        )
    )

    /**
     * Normalisation scale — one unit in each feature dimension that
     * represents a clinically meaningful step, preventing ms-scale features
     * from overwhelming fraction-scale features.
     */
    private val SCALE = doubleArrayOf(50.0, 0.08, 250.0, 0.08, 0.07, 0.07)

    // ── Public result types ────────────────────────────────────────────────

    data class FeatureSummary(
        val meanHold: Double,
        val cvHold: Double,
        val meanIKI: Double,
        val cvIKI: Double,
        val errorRate: Double,
        val longPauseRate: Double,
        val sampleSize: Int
    )

    data class Prediction(
        val label: String,
        val displayName: String,
        val confidence: Double,           // 0.0–1.0
        val features: FeatureSummary,
        val featureFlags: List<FeatureFlag>,
        val allScores: List<Pair<String, Double>>,
        val interpretation: String
    )

    data class FeatureFlag(
        val name: String,
        val value: String,
        val status: Status   // OK / WARNING / ALERT
    ) {
        enum class Status { OK, WARNING, ALERT }
    }

    // ── Entry point ────────────────────────────────────────────────────────

    const val MIN_KEYSTROKES = 20

    /** Returns null if there are fewer than MIN_KEYSTROKES usable events. */
    fun predict(keystrokes: List<KeystrokeEntity>): Prediction? {
        val features = extractFeatures(keystrokes) ?: return null
        val fv = features.toVector()

        val distances = CENTROIDS.map { it to normalisedDistance(fv, it.features) }
        val winner = distances.minByOrNull { it.second }!!.first

        // Softmax-style confidence from inverse distances
        val invSum = distances.sumOf { 1.0 / (it.second + 1e-9) }
        val allScores = distances.map { (c, d) ->
            c.displayName to ((1.0 / (d + 1e-9)) / invSum)
        }
        val winnerConf = allScores.first { it.first == winner.displayName }.second

        return Prediction(
            label = winner.label,
            displayName = winner.displayName,
            confidence = winnerConf,
            features = features,
            featureFlags = buildFlags(features),
            allScores = allScores.sortedByDescending { it.second },
            interpretation = buildInterpretation(features, winner.label)
        )
    }

    // ── Feature extraction ─────────────────────────────────────────────────

    fun extractFeatures(keystrokes: List<KeystrokeEntity>): FeatureSummary? {
        val normal = keystrokes.filter { !it.isBackspace }
        if (normal.size < MIN_KEYSTROKES) return null

        val holds = normal.map { it.holdDuration.toDouble() }
        val meanHold = holds.average()
        val cvHold = standardDeviation(holds) / meanHold.coerceAtLeast(1.0)

        val ikis = normal.map { it.interKeyInterval.toDouble() }.filter { it > 0 }
        val meanIKI = if (ikis.isEmpty()) 0.0 else ikis.average()
        val cvIKI = if (ikis.isEmpty()) 0.0 else
            standardDeviation(ikis) / meanIKI.coerceAtLeast(1.0)

        val errorRate = keystrokes.count { it.isBackspace }.toDouble() /
                keystrokes.size.toDouble()

        val longPauseRate = if (ikis.isEmpty()) 0.0 else
            ikis.count { it > 1500.0 }.toDouble() / ikis.size

        return FeatureSummary(meanHold, cvHold, meanIKI, cvIKI, errorRate, longPauseRate, normal.size)
    }

    private fun FeatureSummary.toVector() =
        doubleArrayOf(meanHold, cvHold, meanIKI, cvIKI, errorRate, longPauseRate)

    // ── Feature flags for UI ───────────────────────────────────────────────

    private fun buildFlags(f: FeatureSummary): List<FeatureFlag> = listOf(
        FeatureFlag(
            name = "Key hold time",
            value = "${"%.0f".format(f.meanHold)} ms",
            status = when {
                f.meanHold > 150 -> FeatureFlag.Status.ALERT
                f.meanHold > 110 -> FeatureFlag.Status.WARNING
                else             -> FeatureFlag.Status.OK
            }
        ),
        FeatureFlag(
            name = "Hold consistency",
            value = "CV ${"%.2f".format(f.cvHold)}",
            status = when {
                f.cvHold > 0.35 -> FeatureFlag.Status.ALERT
                f.cvHold > 0.28 -> FeatureFlag.Status.WARNING
                else            -> FeatureFlag.Status.OK
            }
        ),
        FeatureFlag(
            name = "Key interval",
            value = "${"%.0f".format(f.meanIKI)} ms",
            status = when {
                f.meanIKI > 600 -> FeatureFlag.Status.ALERT
                f.meanIKI > 380 -> FeatureFlag.Status.WARNING
                else            -> FeatureFlag.Status.OK
            }
        ),
        FeatureFlag(
            name = "Rhythm regularity",
            value = "CV ${"%.2f".format(f.cvIKI)}",
            status = when {
                f.cvIKI > 0.33 -> FeatureFlag.Status.ALERT
                f.cvIKI > 0.26 -> FeatureFlag.Status.WARNING
                else           -> FeatureFlag.Status.OK
            }
        ),
        FeatureFlag(
            name = "Error rate",
            value = "${"%.1f".format(f.errorRate * 100)}%",
            status = when {
                f.errorRate > 0.13 -> FeatureFlag.Status.ALERT
                f.errorRate > 0.07 -> FeatureFlag.Status.WARNING
                else               -> FeatureFlag.Status.OK
            }
        ),
        FeatureFlag(
            name = "Long pauses",
            value = "${"%.1f".format(f.longPauseRate * 100)}%",
            status = when {
                f.longPauseRate > 0.12 -> FeatureFlag.Status.ALERT
                f.longPauseRate > 0.06 -> FeatureFlag.Status.WARNING
                else                   -> FeatureFlag.Status.OK
            }
        )
    )

    // ── Interpretation text ────────────────────────────────────────────────

    private fun buildInterpretation(f: FeatureSummary, label: String): String {
        val lines = mutableListOf<String>()

        lines += when {
            f.meanHold > 150 -> "Keys are held for an average of ${"%.0f".format(f.meanHold)} ms — notably longer than the healthy range (70–100 ms). This can reflect motor slowing associated with cognitive decline."
            f.meanHold > 110 -> "Average hold time (${"%.0f".format(f.meanHold)} ms) is moderately elevated above the healthy baseline."
            else             -> "Average hold time (${"%.0f".format(f.meanHold)} ms) is within the normal range."
        }

        lines += when {
            f.meanIKI > 600 -> "Pauses between keystrokes average ${"%.0f".format(f.meanIKI)} ms — substantially longer than typical, suggesting word-finding difficulty or processing delays."
            f.meanIKI > 380 -> "Inter-key pauses (${"%.0f".format(f.meanIKI)} ms average) are moderately elevated."
            else            -> "Typing rhythm (${"%.0f".format(f.meanIKI)} ms average pause) is within the normal range."
        }

        lines += when {
            f.errorRate > 0.13 -> "The error correction rate (${"%.1f".format(f.errorRate * 100)}%) is high, which may reflect difficulty with intended words or motor control."
            f.errorRate > 0.07 -> "Error rate (${"%.1f".format(f.errorRate * 100)}%) is slightly above typical."
            else               -> "Error rate (${"%.1f".format(f.errorRate * 100)}%) is normal."
        }

        if (f.cvIKI > 0.30) {
            lines += "Typing rhythm shows high variability (CV = ${"%.2f".format(f.cvIKI)}), which can indicate inconsistent cognitive processing speed."
        }

        if (label == "early_mci") {
            lines += "\nPre-dementia patterns detected. Early MCI is the stage most responsive to intervention. Regular monitoring is recommended."
        }

        return lines.joinToString("\n\n")
    }

    // ── Maths ──────────────────────────────────────────────────────────────

    private fun normalisedDistance(a: DoubleArray, b: DoubleArray): Double {
        var sum = 0.0
        for (i in a.indices) {
            val diff = (a[i] - b[i]) / SCALE[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }

    private fun standardDeviation(values: List<Double>): Double {
        if (values.size < 2) return 0.0
        val mean = values.average()
        return sqrt(values.sumOf { (it - mean) * (it - mean) } / (values.size - 1))
    }
}