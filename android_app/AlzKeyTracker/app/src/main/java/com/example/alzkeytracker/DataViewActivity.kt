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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
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

        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = "Raw Keystroke Data"

        db = KeystrokeDatabase.getInstance(this)

        binding.recyclerView.layoutManager = LinearLayoutManager(this)
        binding.recyclerView.adapter = adapter

        lifecycleScope.launch {
            db.keystrokeDao().getRecentKeystrokes().collectLatest { keystrokes ->
                adapter.submitList(keystrokes)
                binding.tvCount.text = "Showing ${keystrokes.size} most recent events"
            }
        }

        lifecycleScope.launch {
            val avgHold = withContext(Dispatchers.IO) { db.keystrokeDao().getAverageHoldDuration() }
            val avgIki  = withContext(Dispatchers.IO) { db.keystrokeDao().getAverageIKI() }
            binding.tvStats.text =
                "Avg hold: ${"%.1f".format(avgHold ?: 0.0)} ms   " +
                        "Avg interval: ${"%.1f".format(avgIki ?: 0.0)} ms"
        }

        binding.btnClear.setOnClickListener {
            lifecycleScope.launch(Dispatchers.IO) {
                db.keystrokeDao().deleteAll()
                db.analysisResultDao().deleteAll()
            }
        }
    }

    override fun onSupportNavigateUp(): Boolean { finish(); return true }

    inner class KeystrokeAdapter : RecyclerView.Adapter<KeystrokeAdapter.VH>() {
        private var items: List<KeystrokeEntity> = emptyList()

        fun submitList(list: List<KeystrokeEntity>) { items = list; notifyDataSetChanged() }

        inner class VH(view: View) : RecyclerView.ViewHolder(view) {
            val tvKey: TextView    = view.findViewById(R.id.tv_key)
            val tvTiming: TextView = view.findViewById(R.id.tv_timing)
            val tvTime: TextView   = view.findViewById(R.id.tv_time)
        }

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
            val v = LayoutInflater.from(parent.context)
                .inflate(R.layout.item_keystroke, parent, false)
            return VH(v)
        }

        override fun onBindViewHolder(holder: VH, position: Int) {
            val item = items[position]
            val fmt = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
            holder.tvKey.text = when {
                item.isBackspace        -> "⌫"
                item.keyChar == "SPACE" -> "␣"
                item.keyChar == "ENTER" -> "↵"
                else                   -> item.keyChar
            }
            holder.tvTiming.text =
                "hold ${item.holdDuration} ms  ·  iki ${item.interKeyInterval} ms"
            holder.tvTime.text = fmt.format(Date(item.pressTime))
        }

        override fun getItemCount() = items.size
    }
}