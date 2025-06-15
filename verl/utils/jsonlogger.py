import json
import os
import matplotlib.pyplot as plt
import wandb


class JSONLogger:
    def __init__(self, project: str=None, experiment_name: str=None, config: dict=None):
        base_dir = os.getenv("base_dir", "tmp/logs")
        self.project = project or os.getenv("project", "default_project")
        self.experiment_name = experiment_name or os.getenv("experiment_name", "default_experiment")
        self.dir_path = f"{base_dir}/{self.experiment_name}"
        print(f"JSONLogger: {self.dir_path}")
        self.file_path = os.path.join(self.dir_path, 'logs.json')
        self.config = {}
        self.logs = {}
        if config is None:
            self.read()
        else:
            self.config = config
            self.save_config()

    def read(self):
        if not os.path.exists(self.file_path):
            print(f"File not found: {self.file_path}")
            return
        with open(os.path.join(self.dir_path, 'config.json'), 'r') as f:
            self.config = json.load(f)
        with open(self.file_path, 'r') as f:
            self.logs = json.load(f)

    def save_config(self):
        os.makedirs(self.dir_path, exist_ok=True)
        with open(os.path.join(self.dir_path, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)

    def log(self, data, step=None):
        # 从 0 开始
        self.save_config()
        if step is None:
            # 如果未指定步数，使用当前日志的长度作为步数
            step = len(next(iter(self.logs.values())) if self.logs else [])

        for key, value in data.items():
            if key not in self.logs:
                self.logs[key] = []
            if len(self.logs[key]) <= step:
                self.logs[key].extend([None] * (step - len(self.logs[key]) + 1))
            self.logs[key][step] = value

        with open(self.file_path, 'w') as f:
            json.dump(self.logs, f, indent=4)

    def show(self):
        png_path = os.path.join(self.dir_path, 'logs.png')

        # 按第一个 / 前面的部分对键进行分类
        categories = {}
        for key in self.logs.keys():
            category = key.split('/', 1)[0] if '/' in key else 'no_category'
            if category not in categories:
                categories[category] = []
            categories[category].append(key)

        num_rows = len(categories)
        max_num_cols = max(len(keys) for keys in categories.values())

        fig, axes = plt.subplots(num_rows, max_num_cols, figsize=(5 * max_num_cols, 5 * num_rows))

        if num_rows == 1:
            axes = [axes]

        for i, (category, keys) in enumerate(categories.items()):
            for j, key in enumerate(keys):
                values = self.logs[key]
                steps = list(range(len(values)))
                valid_steps = [step for step, value in zip(steps, values) if value is not None]
                valid_values = [value for value in values if value is not None]
                axes[i][j].plot(valid_steps, valid_values, label=key)
                axes[i][j].set_title(key)
                axes[i][j].set_xlabel('Step')
                axes[i][j].set_ylabel(key)
                axes[i][j].xaxis.get_major_locator().set_params(integer=True)
                axes[i][j].legend()

            # 隐藏多余的子图
            for j in range(len(keys), max_num_cols):
                axes[i][j].axis('off')

        plt.tight_layout()
        print(f"Saving plot to {png_path}")
        plt.savefig(png_path)

    def upload_to_wandb(self):
        wandb.init(project=self.project, name=f"{self.experiment_name}", config=self.config)
        num_steps = max(len(v) for v in self.logs.values())
        for step in range(num_steps):
            log_data = {key: self.logs[key][step] for key in self.logs if step < len(self.logs[key]) and self.logs[key][step] is not None}
            wandb.log(log_data, step=step)
        wandb.finish()


if __name__ == "__main__":
    logger = JSONLogger()

    # logger.log({"step": 0, "loss": 0.5, "accuracy": 0.8}, step=0)
    # logger.log({"step": 1, "loss": 0.3, "accuracy": 0.9}, step=1)

    logger.show()
    if not os.path.isfile("../../mv.sh"):
        logger.upload_to_wandb()
