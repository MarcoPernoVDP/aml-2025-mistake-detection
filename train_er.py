import wandb
from base import fetch_model_name, train_step_test_step_dataset_base, train_sub_step_test_step_dataset_base, \
    train_model_base
from core.config import Config
from core.utils import init_logger_and_wandb
from constants import Constants as const


def train_sub_step_test_step_er(config):
    train_loader, val_loader, test_loader = train_sub_step_test_step_dataset_base(config)
    train_model_base(train_loader, val_loader, config)

def inspect_first_batch(train_loader, val_loader, test_loader):
    loaders = {
        "Train Loader": train_loader,
        "Validation Loader": val_loader,
        "Test Loader": test_loader
    }

    for name, loader in loaders.items():
        print(f"\n=== Inspecting first batch of {name} ===")
        try:
            batch = next(iter(loader))  # Prende solo il primo batch
            step_features, step_labels = batch
            print(f"Step Features: shape={step_features.shape}, dtype={step_features.dtype}")
            print(f"Step Labels: shape={step_labels.shape}, dtype={step_labels.dtype}")
            
            # Stampiamo anche i primi 5 valori per avere un'idea del contenuto
            print("Step Features sample:", step_features[:5])
            print("Step Labels sample:", step_labels[:5])
        except Exception as e:
            print(f"Errore durante l'ispezione di {name}: {e}")

def train_step_test_step_er(config):
    train_loader, val_loader, test_loader = train_step_test_step_dataset_base(config)
    inspect_first_batch(train_loader, val_loader, test_loader)

    #train_model_base(train_loader, val_loader, config, test_loader=test_loader)


def main():
    conf = Config()
    conf.task_name = const.ERROR_RECOGNITION
    if conf.model_name is None:
        m_name = fetch_model_name(conf)
        conf.model_name = m_name

    if conf.enable_wandb:
        init_logger_and_wandb(conf)

    train_step_test_step_er(conf)

    if conf.enable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
