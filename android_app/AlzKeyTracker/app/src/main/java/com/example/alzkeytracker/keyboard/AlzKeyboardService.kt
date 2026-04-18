package com.example.alzkeytracker.keyboard

import android.inputmethodservice.InputMethodService
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.text.InputType
import android.view.KeyEvent
import android.view.View
import android.view.inputmethod.EditorInfo
import android.widget.Button
import androidx.annotation.RequiresApi
import com.example.alzkeytracker.R
import com.example.alzkeytracker.database.KeystrokeDatabase
import com.example.alzkeytracker.database.KeystrokeEntity
import com.example.alzkeytracker.utils.PreferencesManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.util.UUID

class AlzKeyboardService : InputMethodService() {

    private lateinit var prefs: PreferencesManager
    private lateinit var db: KeystrokeDatabase
    private val scope = CoroutineScope(Dispatchers.IO)

    // Session tracking
    private var sessionId: String = UUID.randomUUID().toString()
    private var lastKeyReleaseTime: Long = 0L

    // Shift state
    private var isShifted = false
    private var isCapsLock = false

    // Reference to keyboard view
    private lateinit var keyboardView: View

    override fun onCreate() {
        super.onCreate()
        prefs = PreferencesManager(applicationContext)
        db = KeystrokeDatabase.getInstance(applicationContext)
    }

    override fun onStartInput(attribute: EditorInfo?, restarting: Boolean) {
        super.onStartInput(attribute, restarting)
        // New input field = new session
        if (!restarting) {
            sessionId = UUID.randomUUID().toString()
            lastKeyReleaseTime = 0L
        }
    }

    /**
     * This is called when Android needs to show our keyboard.
     * We inflate our custom XML layout here.
     */
    override fun onCreateInputView(): View {
        keyboardView = layoutInflater.inflate(R.layout.keyboard_view, null)
        setupAllKeys()
        return keyboardView
    }

    @RequiresApi(Build.VERSION_CODES.P)
    private fun setupAllKeys() {
        // ---- Letter rows ----
        val letterRows = listOf(
            listOf("q","w","e","r","t","y","u","i","o","p"),
            listOf("a","s","d","f","g","h","j","k","l"),
            listOf("z","x","c","v","b","n","m")
        )
        val keyIds = listOf(
            // Row 1
            listOf(R.id.key_q, R.id.key_w, R.id.key_e, R.id.key_r, R.id.key_t,
                R.id.key_y, R.id.key_u, R.id.key_i, R.id.key_o, R.id.key_p),
            // Row 2
            listOf(R.id.key_a, R.id.key_s, R.id.key_d, R.id.key_f, R.id.key_g,
                R.id.key_h, R.id.key_j, R.id.key_k, R.id.key_l),
            // Row 3
            listOf(R.id.key_z, R.id.key_x, R.id.key_c, R.id.key_v,
                R.id.key_b, R.id.key_n, R.id.key_m)
        )

        for (rowIndex in letterRows.indices) {
            for (colIndex in letterRows[rowIndex].indices) {
                val char = letterRows[rowIndex][colIndex]
                val button = keyboardView.findViewById<Button>(keyIds[rowIndex][colIndex])
                button.setOnTouchListener(createLetterTouchListener(char, button))
            }
        }

        // ---- Number row ----
        val numChars = listOf("1","2","3","4","5","6","7","8","9","0")
        val numIds = listOf(R.id.key_1, R.id.key_2, R.id.key_3, R.id.key_4, R.id.key_5,
            R.id.key_6, R.id.key_7, R.id.key_8, R.id.key_9, R.id.key_0)
        for (i in numChars.indices) {
            val button = keyboardView.findViewById<Button>(numIds[i])
            button.setOnTouchListener(createLetterTouchListener(numChars[i], button))
        }

        // ---- Special keys ----
        setupSpecialKey(R.id.key_backspace, "BACKSPACE")
        setupSpecialKey(R.id.key_space, "SPACE")
        setupSpecialKey(R.id.key_enter, "ENTER")
        setupSpecialKey(R.id.key_period, ".")
        setupSpecialKey(R.id.key_comma, ",")

        // Shift key
        keyboardView.findViewById<Button>(R.id.key_shift).setOnClickListener {
            toggleShift()
        }

        // Switch keyboard button — brings up the system keyboard picker
        keyboardView.findViewById<Button>(R.id.key_switch_kb).setOnClickListener {
            switchToNextInputMethod(false)
        }
    }

    /**
     * Creates a touch listener that measures EXACT press and release times.
     * This is the core data collection mechanism.
     */
    private fun createLetterTouchListener(
        char: String,
        button: Button
    ): View.OnTouchListener {
        return View.OnTouchListener { _, event ->
            when (event.action) {
                android.view.MotionEvent.ACTION_DOWN -> {
                    val pressTime = System.currentTimeMillis()
                    button.tag = pressTime   // store press time in the button's tag
                    vibrate()
                    true
                }
                android.view.MotionEvent.ACTION_UP -> {
                    val releaseTime = System.currentTimeMillis()
                    val pressTime = (button.tag as? Long) ?: releaseTime
                    val holdDuration = releaseTime - pressTime
                    val iki = if (lastKeyReleaseTime > 0) releaseTime - lastKeyReleaseTime else 0L

                    // Type the character
                    val typedChar = if (isShifted || isCapsLock) char.uppercase() else char
                    typeCharacter(typedChar)

                    // Log to database
                    if (prefs.isLoggingEnabled) {
                        logKeystroke(
                            keyChar = typedChar,
                            pressTime = pressTime,
                            releaseTime = releaseTime,
                            holdDuration = holdDuration,
                            iki = iki,
                            isBackspace = false
                        )
                    }

                    lastKeyReleaseTime = releaseTime

                    // Reset shift (but not caps lock)
                    if (isShifted && !isCapsLock) {
                        isShifted = false
                        updateShiftVisuals()
                    }
                    true
                }
                else -> false
            }
        }
    }

    private fun setupSpecialKey(keyId: Int, keyName: String) {
        val button = keyboardView.findViewById<Button>(keyId)
        button.setOnTouchListener { _, event ->
            when (event.action) {
                android.view.MotionEvent.ACTION_DOWN -> {
                    button.tag = System.currentTimeMillis()
                    vibrate()
                    true
                }
                android.view.MotionEvent.ACTION_UP -> {
                    val releaseTime = System.currentTimeMillis()
                    val pressTime = (button.tag as? Long) ?: releaseTime
                    val holdDuration = releaseTime - pressTime
                    val iki = if (lastKeyReleaseTime > 0) releaseTime - lastKeyReleaseTime else 0L

                    when (keyName) {
                        "BACKSPACE" -> {
                            currentInputConnection?.deleteSurroundingText(1, 0)
                            if (prefs.isLoggingEnabled) {
                                logKeystroke("BACKSPACE", pressTime, releaseTime, holdDuration, iki, true)
                            }
                        }
                        "SPACE" -> {
                            typeCharacter(" ")
                            if (prefs.isLoggingEnabled) {
                                logKeystroke("SPACE", pressTime, releaseTime, holdDuration, iki, false)
                            }
                        }
                        "ENTER" -> {
                            currentInputConnection?.sendKeyEvent(
                                KeyEvent(KeyEvent.ACTION_DOWN, KeyEvent.KEYCODE_ENTER)
                            )
                            currentInputConnection?.sendKeyEvent(
                                KeyEvent(KeyEvent.ACTION_UP, KeyEvent.KEYCODE_ENTER)
                            )
                            if (prefs.isLoggingEnabled) {
                                logKeystroke("ENTER", pressTime, releaseTime, holdDuration, iki, false)
                            }
                        }
                        else -> {
                            typeCharacter(keyName)
                            if (prefs.isLoggingEnabled) {
                                logKeystroke(keyName, pressTime, releaseTime, holdDuration, iki, false)
                            }
                        }
                    }

                    lastKeyReleaseTime = releaseTime
                    true
                }
                else -> false
            }
        }
    }

    private fun typeCharacter(char: String) {
        currentInputConnection?.commitText(char, 1)
    }

    private fun logKeystroke(
        keyChar: String,
        pressTime: Long,
        releaseTime: Long,
        holdDuration: Long,
        iki: Long,
        isBackspace: Boolean
    ) {
        val appPackage = currentInputEditorInfo?.packageName ?: "unknown"
        scope.launch {
            db.keystrokeDao().insert(
                KeystrokeEntity(
                    sessionId = sessionId,
                    keyChar = keyChar,
                    pressTime = pressTime,
                    releaseTime = releaseTime,
                    holdDuration = holdDuration,
                    interKeyInterval = iki,
                    isBackspace = isBackspace,
                    appPackage = appPackage,
                    userId = prefs.userId,
                    syntheticLabel = "real"
                )
            )
        }
    }

    private fun toggleShift() {
        when {
            !isShifted && !isCapsLock -> {
                isShifted = true
                isCapsLock = false
            }
            isShifted && !isCapsLock -> {
                // Double-tap = caps lock
                isCapsLock = true
                isShifted = false
            }
            isCapsLock -> {
                isCapsLock = false
                isShifted = false
            }
        }
        updateShiftVisuals()
    }

    private fun updateShiftVisuals() {
        val shiftBtn = keyboardView.findViewById<Button>(R.id.key_shift)
        shiftBtn.text = when {
            isCapsLock -> "⬆⬆"
            isShifted -> "⬆"
            else -> "⇧"
        }
        // Update letter key labels to show upper/lowercase
        updateLetterLabels()
    }

    private fun updateLetterLabels() {
        val letters = mapOf(
            R.id.key_q to "q", R.id.key_w to "w", R.id.key_e to "e",
            R.id.key_r to "r", R.id.key_t to "t", R.id.key_y to "y",
            R.id.key_u to "u", R.id.key_i to "i", R.id.key_o to "o",
            R.id.key_p to "p", R.id.key_a to "a", R.id.key_s to "s",
            R.id.key_d to "d", R.id.key_f to "f", R.id.key_g to "g",
            R.id.key_h to "h", R.id.key_j to "j", R.id.key_k to "k",
            R.id.key_l to "l", R.id.key_z to "z", R.id.key_x to "x",
            R.id.key_c to "c", R.id.key_v to "v", R.id.key_b to "b",
            R.id.key_n to "n", R.id.key_m to "m"
        )
        val useUpper = isShifted || isCapsLock
        for ((id, char) in letters) {
            keyboardView.findViewById<Button>(id).text = if (useUpper) char.uppercase() else char
        }
    }

    private fun vibrate() {
        val vibrator = getSystemService(VIBRATOR_SERVICE) as? Vibrator
        vibrator?.vibrate(VibrationEffect.createOneShot(20, VibrationEffect.DEFAULT_AMPLITUDE))
    }
}