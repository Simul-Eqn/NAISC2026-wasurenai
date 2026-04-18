package com.example.alzkeytracker

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.alzkeytracker.database.KeystrokeDatabase
import com.example.alzkeytracker.database.KeystrokeEntity
import com.example.alzkeytracker.databinding.ActivityDataViewBinding
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class DataViewActivity : AppCompatActivity() {

    private lateinit var binding: ActivityDataViewBinding
    private lateinit var db: KeystrokeDatabase
    private val adapter = KeystrokeAdapter()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityDataViewBinding.inflate(layoutInflater)
        setContentView(binding.root)

        db = KeystrokeDatabase.getInstance(this)

        binding.recyclerView.layoutManager = LinearLayoutManager(this)
        binding.recyclerView.adapter = adapter

        // Observe live data
        lifecycleScope.launch {
            db.keystrokeDao().getRecentKeystrokes().collect { keystrokes ->
                adapter.submitList(keystrokes)
                binding.tvCount.text = "Showing ${keystrokes.size} most recent events"
            }
        }

        // Stats
        lifecycleScope.launch {
            val avgHold = db.keystrokeDao().getAverageHoldDuration()
            val avgIki = db.keystrokeDao().getAverageIKI()
            val backspaceCount = db.keystrokeDao().getBackspaceCount()
            val total = db.keystrokeDao().getTotalCount()
            val errorRate = if (total > 0) (backspaceCount.toFloat() / total * 100) else 0f

            runOnUiThread {
                binding.tvStats.text = """
                    📊 Summary Statistics
                    Avg hold duration: ${"%.1f".format(avgHold ?: 0.0)} ms
                    Avg inter-key interval: ${"%.1f".format(avgIki ?: 0.0)} ms
                    Error rate (backspace %): ${"%.1f".format(errorRate)}%
                    
                    ℹ️ Alzheimer's indicators:
                    • Hold > 150ms → possible motor slowing
                    • IKI > 500ms → possible hesitation
                    • Error rate > 10% → possible cognitive load
                """.trimIndent()
            }
        }

        binding.btnClearData.setOnClickListener {
            lifecycleScope.launch {
                db.keystrokeDao().deleteAll()
            }
        }

        supportActionBar?.setDisplayHomeAsUpEnabled(true)
    }

    override fun onSupportNavigateUp(): Boolean {
        finish()
        return true
    }
}

// ---- RecyclerView Adapter ----

class KeystrokeAdapter : RecyclerView.Adapter<KeystrokeAdapter.VH>() {

    private var items: List<KeystrokeEntity> = emptyList()

    fun submitList(list: List<KeystrokeEntity>) {
        items = list
        notifyDataSetChanged()
    }

    inner class VH(view: View) : RecyclerView.ViewHolder(view) {
        val tvKey: TextView = view.findViewById(R.id.tv_key)
        val tvTiming: TextView = view.findViewById(R.id.tv_timing)
        val tvLabel: TextView = view.findViewById(R.id.tv_label)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_keystroke, parent, false)
        return VH(view)
    }

    override fun onBindViewHolder(holder: VH, position: Int) {
        val item = items[position]
        val fmt = SimpleDateFormat("HH:mm:ss.SSS", Locale.getDefault())
        holder.tvKey.text = if (item.isBackspace) "⌫ DEL" else "\"${item.keyChar}\""
        holder.tvTiming.text = "Hold: ${item.holdDuration}ms  IKI: ${item.interKeyInterval}ms  " +
                "@${fmt.format(Date(item.pressTime))}"
        holder.tvLabel.text = "[${item.syntheticLabel}] ${item.appPackage.takeLast(20)}"
    }

    override fun getItemCount() = items.size
}