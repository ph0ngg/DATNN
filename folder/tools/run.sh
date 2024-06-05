#!/bin/bash
# Execute the four Python files in order
python tools/test.py --config-file /home/phongnn/test/test/fast-reid/configs/Market1501/AGW_R50.yml

python tools/reid.py

python tools/get_reid_txt_file.py

python tools/visualize.py

echo "All Python files executed in order!"