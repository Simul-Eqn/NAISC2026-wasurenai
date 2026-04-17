I think i got maybe 1000 data points; but a bit limited because 

```
... 
860
880
900
920
940
960
980
Traceback (most recent call last):
  File "D:\Simu\projects\2026\NAISC2026-wasurenai\new_inference_server\voxpopuli\run_meralion.py", line 189, in run
    json.dump(payload, file, ensure_ascii=False)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Simu\conda_envs\naisc26_meralion\Lib\json\__init__.py", line 182, in dump
    fp.write(chunk)
    ~~~~~~~~^^^^^^^
OSError: [Errno 28] No space left on device

During handling of the above exception, another exception occurred:

OSError: [Errno 28] No space left on device

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "D:\Simu\projects\2026\NAISC2026-wasurenai\new_inference_server\voxpopuli\run_meralion.py", line 254, in <module>
    run()
    ~~~^^
  File "D:\Simu\projects\2026\NAISC2026-wasurenai\new_inference_server\voxpopuli\run_meralion.py", line 188, in run
    with output_file.open("w", encoding="utf-8") as file:
         ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
OSError: [Errno 28] No space left on device
```
