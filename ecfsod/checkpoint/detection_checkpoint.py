import pickle
import detectron2.utils.comm as comm
from fvcore.common.file_io import PathManager
from fvcore.common.checkpoint import Checkpointer
from detectron2.checkpoint.c2_model_loading import align_and_update_state_dicts

class DetectionCheckpointer(Checkpointer):
    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        is_main_process = comm.is_main_process()
        super().__init__(
            model,
            save_dir,
            save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
            **checkpointables,
        )

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                if "blobs" in data:
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}
        loaded = super()._load_file(filename)
        if "model" not in loaded:
            loaded = {"model": loaded}
        return loaded

    def _load_model(self, checkpoint):
        if checkpoint.get("matching_heuristics", False):
            self._convert_ndarray_to_tensor(checkpoint["model"])
            model_state_dict = self.model.state_dict()
            align_and_update_state_dicts(
                model_state_dict,
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )
            checkpoint["model"] = model_state_dict
        super()._load_model(checkpoint)