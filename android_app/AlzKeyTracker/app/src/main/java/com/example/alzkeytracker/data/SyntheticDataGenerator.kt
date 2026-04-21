package com.example.alzkeytracker.data

import com.example.alzkeytracker.database.KeystrokeEntity
import java.util.UUID
import kotlin.math.max
import kotlin.random.Random

/**
 * Generates synthetic keystroke data that mimics real clinical patterns.
 *
 * Key biomarkers modelled:
 * - Hold Duration: how long each key is pressed
 * - Inter-Key Interval (IKI): pause between keystrokes
 * - Error Rate: frequency of backspace/delete
 * - Typing Speed: words per minute equivalent
 *
 * Sources for parameter ranges:
 * - Arroyo-Gallego et al. (2017) "Detection of Motor Impairment in Parkinson's Disease"
 * - Garn et al. (2019) keystroke dynamics in MCI patients
 */
object SyntheticDataGenerator {

    // Sentence pool — realistic everyday messages elderly people might type
    private val sentencePool = listOf(
        "hello how are you doing today",
        "i will call you later this evening",
        "please remember to take your medicine",
        "the weather is nice outside today",
        "i am feeling a bit tired",
        "can you come visit me this weekend",
        "i had a good breakfast this morning",
        "the doctor said everything looks fine",
        "my grandson came to visit yesterday",
        "i need to go to the pharmacy",
        "did you watch the news last night",
        "i cannot find my reading glasses",
        "the garden looks beautiful this time of year",
        "please bring some fruit when you come",
        "i forgot what i was going to say"
    )

    /**
     * Profile defines the statistical parameters for each group.
     * All times in milliseconds.
     */
    data class TypingProfile(
        val label: String,
        // Mean hold duration (ms) — how long a key is pressed
        val holdMean: Double,
        val holdStd: Double,
        // Mean inter-key interval (ms) — pause between keys
        val ikiMean: Double,
        val ikiStd: Double,
        // Probability of a backspace event after any character
        val errorProbability: Double,
        // Probability of a long pause (>2s) in the middle of typing
        val pauseProbability: Double,
        // Mean duration of those long pauses (ms)
        val pauseDurationMean: Double
    )

    // === PROFILE PARAMETERS ===
    // Based on literature values, scaled to realistic ranges

    private val healthyProfile = TypingProfile(
        label = "healthy",
        holdMean = 85.0,       // ~85ms hold — normal elderly typist
        holdStd = 18.0,
        ikiMean = 250.0,       // ~250ms between keys — moderate pace
        ikiStd = 55.0,
        errorProbability = 0.04,   // 4% of keystrokes followed by backspace
        pauseProbability = 0.02,   // 2% chance of a long pause between words
        pauseDurationMean = 1200.0
    )

    private val earlyMCIProfile = TypingProfile(
        label = "early_mci",
        holdMean = 115.0,      // Slightly longer press times (motor slowdown)
        holdStd = 35.0,        // More variance
        ikiMean = 420.0,       // Longer pauses between keys
        ikiStd = 120.0,        // Much higher variance — inconsistency is key
        errorProbability = 0.09,   // 9% error rate
        pauseProbability = 0.08,
        pauseDurationMean = 2800.0
    )

    private val moderateADProfile = TypingProfile(
        label = "moderate_ad",
        holdMean = 175.0,      // Very long key holds
        holdStd = 65.0,
        ikiMean = 780.0,       // Nearly 0.8s between every keystroke
        ikiStd = 280.0,        // Very high variance — unpredictable pauses
        errorProbability = 0.18,   // 18% error rate
        pauseProbability = 0.18,
        pauseDurationMean = 5500.0
    )

    /**
     * Generate a full synthetic dataset.
     * @param sessionsPerGroup How many typing sessions to simulate per group
     * @param userId The participant/user ID to tag data with
     */
    fun generateDataset(sessionsPerGroup: Int = 5, userId: String = "synthetic_user"): List<KeystrokeEntity> {
        val allData = mutableListOf<KeystrokeEntity>()
        val profiles = listOf(healthyProfile, earlyMCIProfile, moderateADProfile)

        for (profile in profiles) {
            repeat(sessionsPerGroup) {
                val sentence = sentencePool.random()
                allData.addAll(generateSession(sentence, profile, userId))
            }
        }
        return allData
    }

    /**
     * Generate keystroke events for one typing session (one sentence).
     */
    fun generateSession(
        sentence: String,
        profile: TypingProfile,
        userId: String
    ): List<KeystrokeEntity> {
        val sessionId = UUID.randomUUID().toString()
        val events = mutableListOf<KeystrokeEntity>()

        // Session starts at a random time in the past (last 30 days)
        var currentTime = System.currentTimeMillis() -
                Random.nextLong(0, 30L * 24 * 60 * 60 * 1000)

        var lastReleaseTime = currentTime
        var previousReleaseTime = 0L

        for (char in sentence) {
            // Maybe inject a long pause before this character (simulates losing train of thought)
            if (Random.nextDouble() < profile.pauseProbability) {
                val pause = gaussianPositive(profile.pauseDurationMean, profile.pauseDurationMean * 0.4)
                currentTime += pause.toLong()
                lastReleaseTime = currentTime
            }

            // Inter-key interval from last release to this press
            val iki = if (previousReleaseTime == 0L) 0L
            else max(50L, gaussianPositive(profile.ikiMean, profile.ikiStd).toLong())

            val pressTime = lastReleaseTime + iki

            // How long the key is held
            val holdDuration = max(30L, gaussianPositive(profile.holdMean, profile.holdStd).toLong())
            val releaseTime = pressTime + holdDuration

            val keyStr = when (char) {
                ' ' -> "SPACE"
                else -> char.toString()
            }

            events.add(
                KeystrokeEntity(
                    sessionId = sessionId,
                    keyChar = keyStr,
                    pressTime = pressTime,
                    releaseTime = releaseTime,
                    holdDuration = holdDuration,
                    interKeyInterval = iki,
                    isBackspace = false,
                    appPackage = "com.example.alzkeytracker",
                    userId = userId,
                    syntheticLabel = profile.label
                )
            )

            previousReleaseTime = releaseTime
            lastReleaseTime = releaseTime
            currentTime = releaseTime

            // Maybe inject an error (backspace) after this character
            if (char != ' ' && Random.nextDouble() < profile.errorProbability) {
                // Short pause then backspace
                val errorIki = max(80L, gaussianPositive(200.0, 60.0).toLong())
                val bsPressTime = releaseTime + errorIki
                val bsHold = max(40L, gaussianPositive(profile.holdMean * 0.9, profile.holdStd).toLong())
                val bsReleaseTime = bsPressTime + bsHold

                events.add(
                    KeystrokeEntity(
                        sessionId = sessionId,
                        keyChar = "BACKSPACE",
                        pressTime = bsPressTime,
                        releaseTime = bsReleaseTime,
                        holdDuration = bsHold,
                        interKeyInterval = errorIki,
                        isBackspace = true,
                        appPackage = "com.example.alzkeytracker",
                        userId = userId,
                        syntheticLabel = profile.label
                    )
                )

                // Re-type the character after correcting
                val retypeIki = max(80L, gaussianPositive(profile.ikiMean * 0.8, profile.ikiStd).toLong())
                val rtPressTime = bsReleaseTime + retypeIki
                val rtHold = max(30L, gaussianPositive(profile.holdMean, profile.holdStd).toLong())
                val rtReleaseTime = rtPressTime + rtHold

                events.add(
                    KeystrokeEntity(
                        sessionId = sessionId,
                        keyChar = keyStr,
                        pressTime = rtPressTime,
                        releaseTime = rtReleaseTime,
                        holdDuration = rtHold,
                        interKeyInterval = retypeIki,
                        isBackspace = false,
                        appPackage = "com.example.alzkeytracker",
                        userId = userId,
                        syntheticLabel = profile.label
                    )
                )

                previousReleaseTime = rtReleaseTime
                lastReleaseTime = rtReleaseTime
                currentTime = rtReleaseTime
            }
        }

        return events
    }

    // Gaussian random that never goes negative
    private fun gaussianPositive(mean: Double, std: Double): Double {
        val value = mean + Random.nextDouble(-1.0, 1.0) * std * 1.732
        return max(1.0, value)
    }
}