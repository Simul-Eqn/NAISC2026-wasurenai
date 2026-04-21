package com.example.alzkeytracker

import android.content.res.ColorStateList
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.alzkeytracker.data.DementiaPredictor
import com.example.alzkeytracker.database.AnalysisResultEntity
import com.example.alzkeytracker.database.KeystrokeDatabase
import com.example.alzkeytracker.databinding.ActivityAnalysisBinding
import com.example.alzkeytracker.utils.PreferencesManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class AnalysisActivity : AppCompatActivity() {

    private lateinit var binding: ActivityAnalysisBinding
    private lateinit var db: KeystrokeDatabase
    private lateinit var prefs: PreferencesManager
    private val historyAdapter = HistoryAdapter()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityAnalysisBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = "Analysis"

        db = KeystrokeDatabase.getInstance(this)
        prefs = PreferencesManager(this)

        binding.recyclerHistory.layoutManager = LinearLayoutManager(this)
        binding.recyclerHistory.adapter = historyAdapter
        binding.recyclerHistory.isNestedScrollingEnabled = false

        runAnalysis()
    }

    override fun onSupportNavigateUp(): Boolean { finish(); return true }

    private fun runAnalysis() {
        binding.progressBar.visibility = View.VISIBLE
        binding.layoutResult.visibility = View.GONE
        binding.layoutInsufficient.visibility = View.GONE

        lifecycleScope.launch {
            val keystrokes = withContext(Dispatchers.IO) {
                db.keystrokeDao().getKeystrokesForUser(prefs.userId)
                    .filter { it.syntheticLabel == "real" }
            }

            val prediction = withContext(Dispatchers.Default) {
                DementiaPredictor.predict(keystrokes)
            }

            binding.progressBar.visibility = View.GONE

            if (prediction == null) {
                val have = keystrokes.filter { !it.isBackspace }.size
                binding.tvInsufficientDetail.text =
                    "Collected $have of ${DementiaPredictor.MIN_KEYSTROKES} required keystrokes.\n\n" +
                            "Type using the Wasurenai keyboard in any app to collect data."
                binding.layoutInsufficient.visibility = View.VISIBLE
                return@launch
            }

            // Save result to progression history
            val entity = AnalysisResultEntity(
                timestamp = System.currentTimeMillis(),
                userId = prefs.userId,
                label = prediction.label,
                displayName = prediction.displayName,
                confidence = prediction.confidence,
                meanHold = prediction.features.meanHold,
                meanIKI = prediction.features.meanIKI,
                errorRate = prediction.features.errorRate,
                longPauseRate = prediction.features.longPauseRate,
                sampleSize = prediction.features.sampleSize
            )
            withContext(Dispatchers.IO) { db.analysisResultDao().insert(entity) }

            // Show result
            displayResult(prediction)

            // Load and show progression
            val history = withContext(Dispatchers.IO) {
                db.analysisResultDao().getAllForUser(prefs.userId)
            }
            displayProgression(history)

            binding.layoutResult.visibility = View.VISIBLE
        }
    }

    private fun displayResult(p: DementiaPredictor.Prediction) {
        val (bgColor, fgColor) = when (p.label) {
            "healthy"      -> R.color.healthy_bg to R.color.healthy_fg
            "early_mci"    -> R.color.mci_bg to R.color.mci_fg
            "moderate_ad"  -> R.color.ad_bg to R.color.ad_fg
            else           -> R.color.healthy_bg to R.color.healthy_fg
        }
        binding.cardResult.setCardBackgroundColor(ContextCompat.getColor(this, bgColor))
        binding.tvResultLabel.setTextColor(ContextCompat.getColor(this, fgColor))
        binding.tvResultLabel.text = p.displayName
        binding.tvResultConfidence.setTextColor(ContextCompat.getColor(this, fgColor))
        binding.tvResultConfidence.text =
            "Confidence ${"%.0f".format(p.confidence * 100)}%  ·  ${p.features.sampleSize} keystrokes"

        // Score breakdown
        binding.tvScoreBreakdown.text = p.allScores.joinToString("   ") { (name, score) ->
            "$name ${"%.0f".format(score * 100)}%"
        }

        // Feature flags
        binding.layoutFlags.removeAllViews()
        for (flag in p.featureFlags) {
            val row = layoutInflater.inflate(
                android.R.layout.simple_list_item_2, binding.layoutFlags, false)
            row.findViewById<TextView>(android.R.id.text1).apply {
                text = "${flag.name}:  ${flag.value}"
                textSize = 13f
                setTextColor(ContextCompat.getColor(this@AnalysisActivity, R.color.text_primary))
            }
            row.findViewById<TextView>(android.R.id.text2).apply {
                text = when (flag.status) {
                    DementiaPredictor.FeatureFlag.Status.OK      -> "✅  Normal"
                    DementiaPredictor.FeatureFlag.Status.WARNING -> "🔶  Elevated"
                    DementiaPredictor.FeatureFlag.Status.ALERT   -> "⚠️  Concerning"
                }
                textSize = 12f
            }
            binding.layoutFlags.addView(row)
        }

        // Interpretation
        binding.tvInterpretation.text = p.interpretation

        // Disclaimer
        binding.tvDisclaimer.text = getString(R.string.disclaimer)
    }

    private fun displayProgression(history: List<AnalysisResultEntity>) {
        if (history.size < 2) {
            binding.tvTrend.text = "Run more analyses over time to see your progression trend."
            historyAdapter.submitList(history)
            return
        }

        binding.tvTrend.text = computeTrend(history)
        historyAdapter.submitList(history)
    }

    /**
     * Compares older vs newer halves of the result history to detect trend.
     * History is newest-first, so we reverse before splitting.
     */
    private fun computeTrend(results: List<AnalysisResultEntity>): String {
        val scores = results.reversed().map {
            when (it.label) { "healthy" -> 0.0; "early_mci" -> 1.0; else -> 2.0 }
        }
        val mid = scores.size / 2
        val older = scores.take(mid).average()
        val newer = scores.drop(mid).average()

        return when {
            newer > older + 0.35 ->
                "📈  Trend: Cognitive markers appear to be increasing — recent sessions show more concern than earlier ones. Consider discussing with a healthcare provider."
            newer < older - 0.35 ->
                "📉  Trend: Positive — recent sessions show patterns closer to the healthy baseline compared to earlier ones."
            else ->
                "📊  Trend: Stable — no significant change detected across ${results.size} sessions."
        }
    }

    // ── Progression history adapter ───────────────────────────────────────

    inner class HistoryAdapter : RecyclerView.Adapter<HistoryAdapter.VH>() {
        private var items: List<AnalysisResultEntity> = emptyList()

        fun submitList(list: List<AnalysisResultEntity>) {
            items = list; notifyDataSetChanged()
        }

        inner class VH(view: View) : RecyclerView.ViewHolder(view) {
            val tvDate: TextView       = view.findViewById(R.id.tv_date)
            val tvDetail: TextView     = view.findViewById(R.id.tv_detail)
            val tvLabel: TextView      = view.findViewById(R.id.tv_label)
        }

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
            val v = LayoutInflater.from(parent.context)
                .inflate(R.layout.item_analysis_result, parent, false)
            return VH(v)
        }

        override fun onBindViewHolder(holder: VH, position: Int) {
            val item = items[position]
            val fmt = SimpleDateFormat("d MMM yyyy  HH:mm", Locale.getDefault())
            holder.tvDate.text   = fmt.format(Date(item.timestamp))
            holder.tvDetail.text =
                "${item.sampleSize} keystrokes · ${"%.0f".format(item.confidence * 100)}% conf"

            val (bg, fg) = when (item.label) {
                "healthy"     -> R.color.healthy_bg to R.color.healthy_fg
                "early_mci"   -> R.color.mci_bg to R.color.mci_fg
                "moderate_ad" -> R.color.ad_bg to R.color.ad_fg
                else          -> R.color.healthy_bg to R.color.healthy_fg
            }
            holder.tvLabel.backgroundTintList =
                ColorStateList.valueOf(ContextCompat.getColor(holder.tvLabel.context, bg))
            holder.tvLabel.setTextColor(ContextCompat.getColor(holder.tvLabel.context, fg))
            holder.tvLabel.text = item.displayName
        }

        override fun getItemCount() = items.size
    }
}