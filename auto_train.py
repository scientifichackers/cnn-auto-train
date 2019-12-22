import json
import typing as T
from functools import wraps
from pathlib import Path
import keras.backend as K
from keras import Model
from keras.callbacks import History, EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from keras_preprocessing.image import np


class DoFit:
    def __call__(self, session: int, model: Model, **kwargs) -> History:
        ...


class InitModel:
    def __call__(self) -> Model:
        ...


def _generator_to_list(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return list(fn(*args, **kwargs))

    return wrapper


@_generator_to_list
def train(
    init_model: InitModel,
    do_fit: DoFit,
    save_dir: T.Union[str, Path],
    *,
    sessions: int = 10,
    epochs: int = 500,
    verbose: int = 2,
    patience: int = 10,
    lr_factor: float = 0.1,
) -> T.List[T.Dict[str, T.List[float]]]:
    save_dir = Path(save_dir)
    model, cur_session, cur_epoch, cur_val_loss = _load_model(
        save_dir, init_model, verbose
    )

    while cur_session < sessions:
        if verbose > 1:
            print(f'\n*** Training Session {cur_session} ***\n')

        session_dir = save_dir / str(cur_session)
        session_dir.mkdir(parents=True, exist_ok=True)

        model_filepath = str(session_dir / '{epoch}.hdf5')

        while cur_epoch < epochs:
            checkpoint = CustomModelCheckpoint(
                model_filepath,
                verbose=verbose,
                save_best_only=True,
                initial_best=cur_val_loss,
            )

            history = do_fit(
                cur_session,
                model,
                callbacks=[
                    checkpoint,
                    EarlyStopping(
                        patience=patience,
                        verbose=verbose,
                        restore_best_weights=True,
                        baseline=cur_val_loss,
                    ),
                ],
                epochs=epochs,
                initial_epoch=cur_epoch,
                verbose=verbose,
            )

            K.set_value(model.optimizer.lr, K.get_value(model.optimizer.lr) * lr_factor)

            cur_epoch = history.epoch[-1] + 1
            cur_val_loss = checkpoint.best

            yield history

        cur_epoch = 0
        cur_session += 1


def _load_model(
    save_dir: Path, init_model: InitModel, verbose: int
) -> T.Tuple[Model, int, int, int]:
    cur_session = 1
    cur_epoch = 0
    cur_val_loss = np.Inf

    model = None
    if save_dir.exists():
        try:
            session_dir, cur_session = _get_most_recent(save_dir)
            model_file, cur_epoch = _get_most_recent(session_dir)
        except ValueError:
            pass
        else:
            logs_file = model_file.parent / (model_file.stem + '.json')
            logs = json.loads(logs_file.read_text())
            cur_val_loss = logs['val_loss']

            model_filepath = str(model_file)
            if verbose > 1:
                print(
                    f'Resumed training from session: {cur_session}, epoch: {cur_epoch}, val_loss: {cur_val_loss}'
                )
                print(f'Loading model @ {model_filepath!r}...')
            model = load_model(model_filepath)
    else:
        save_dir.mkdir(parents=True)
    if model is None:
        model = init_model()

    return model, cur_session, cur_epoch, cur_val_loss


def _get_most_recent(directory: Path) -> T.Tuple[Path, int]:
    sessions = ((i, int(i.stem)) for i in directory.iterdir() if i.stem.isnumeric())
    return max(sessions, key=lambda x: x[1])


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        filepath,
        monitor='val_loss',
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        period=1,
        initial_best=None,
    ):
        super().__init__(
            filepath, monitor, verbose, save_best_only, save_weights_only, mode, period
        )

        if initial_best is not None:
            self.best = initial_best

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        epoch += 1
        file = Path(self.filepath.format(epoch=epoch, **logs))

        # model checkpoint did not save at this epoch, so abort!
        if not file.exists():
            return

        history_file = Path(file.parent / 'history.json')
        logs_file = Path(file.parent / f'{epoch}.json')

        if self.verbose > 1:
            print(f"Saving logs to '{logs_file}' & '{history_file}'...")

        logs_file.write_text(json.dumps(logs))

        if history_file.exists():
            history = json.loads(history_file.read_text())
            for k, v in history.items():
                history[k].append(logs[k])
        else:
            history = {}
            for k, v in logs.items():
                history[k] = [v]
        history_file.write_text(json.dumps(history))
