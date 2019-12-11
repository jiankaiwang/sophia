# VoTT-CNTK WorkFlow



### 依據 VoTT 結果產生 Map 檔案

* 複製 `CNTK-Samples-2-4\Examples\Image\Detection\utils\annotations\annotations_helper.py` 出一份檔案位於相同資料夾，名為 `annotations_helper_args.py`
* 並於檔案 `annotations_helper_args.py` 中修改 **entry** 部分：

```python
if __name__ == '__main__':
    abs_path = os.path.dirname(os.path.abspath(__file__))
    # 預設為路徑在 ../../../DataSets/Grocery 的資料夾
    #data_set_path = os.path.join(abs_path, "../../../DataSets/Grocery")
    
    # 改成 vott 輸出資料集的路徑
    # data_set_path = os.path.join(abs_path, "vott_output")
    data_set_path = "/path/to/your/vott/output/absolute/path"
    
    ...
```

*   執行此份 python Script:

```bash
# 成功會產出底下數筆資料，會於底下組態設定檔案中使用
# class_map.txt
# test_img_file.txt
# test_roi_file.txt
# train_img_file.txt
# train_roi_file.txt
python annotations_helper_args.py
```



### 建立資料集專屬組態

* 複製 `CNTK-Samples-2-4\Examples\Image\Detection\utils\configs\Grocery_config.py` 出一份檔案位於相同資料夾，名為 `MyDataSet_config.py` (檔名可自取)
* 並對檔案 `MyDataSet_config.py` 進行修改，如下:

```python
...

# data set config
__C.DATA.DATASET = "your_dataset_name"
__C.DATA.MAP_FILE_PATH = "/path/to/your/vott/output/absolute/path"
__C.DATA.CLASS_MAP_FILE = "class_map.txt"
__C.DATA.TRAIN_MAP_FILE = "train_img_file.txt"
__C.DATA.TRAIN_ROI_FILE = "train_roi_file.txt"
__C.DATA.TEST_MAP_FILE = "test_img_file.txt"
__C.DATA.TEST_ROI_FILE = "test_roi_file.txt"
__C.DATA.NUM_TRAIN_IMAGES = 20
__C.DATA.NUM_TEST_IMAGES = 5
__C.DATA.PROPOSAL_LAYER_SCALES = [4, 8, 12]

...
```



### 執行深度學習模型

*   舉執行 **Faster R-CNN** (位於 `CNTK-Samples-2-4\Examples\Image\Detection\FasterRCNN\run_faster_rcnn.py`) 為例。
*   複製 `CNTK-Samples-2-4\Examples\Image\Detection\FasterRCNN\run_faster_rcnn.py` 出一份檔案位於相同資料夾，名為 `run_faster_rcnn_args.py`。
*   編輯檔案 `run_faster_rcnn_args.py`，如下:

```python
# ...

def get_configuration():
    # load configs for detector, base network and data set
    from FasterRCNN_config import cfg as detector_cfg
    # for VGG16 base model use: from utils.configs.VGG16_config import cfg as network_cfg
    # for AlexNet base model use: from utils.configs.AlexNet_config import cfg as network_cfg
    from utils.configs.AlexNet_config import cfg as network_cfg
    # for Pascal VOC 2007 data set use: from utils.configs.Pascal_config import cfg as dataset_cfg
    # for the Grocery data set use: from utils.configs.Grocery_config import cfg as dataset_cfg
    #from utils.configs.Grocery_config import cfg as dataset_cfg
    
    # 將剛產生的 MyDataSet_config 匯入
    from utils.configs.MyDataSet_config import cfg as dataset_cfg

    return merge_configs([detector_cfg, network_cfg, dataset_cfg])

# ...
```

*   執行此 **FRCNN** 模型，如下:

```bash
python run_faster_rcnn_args.py
```

