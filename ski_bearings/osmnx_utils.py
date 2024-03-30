import contextlib
import warnings
from collections.abc import Generator


@contextlib.contextmanager
def suppress_user_warning(
    category: type[Warning] = UserWarning, message: str = ""
) -> Generator[None, None, None]:
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message=message, category=category)
        yield
