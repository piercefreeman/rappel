from rappel import action, workflow
from rappel.workflow import Workflow


class CustomError(Exception):
    """Custom error for testing specific exception catching."""
    pass


@action
async def get_error_type() -> str:
    """Returns which error type to raise."""
    return "value_error"  # Can be "value_error", "type_error", or "custom_error"


@action
async def raise_value_error() -> str:
    raise ValueError("value error occurred")


@action
async def raise_type_error() -> str:
    raise TypeError("type error occurred")


@action
async def raise_custom_error() -> str:
    raise CustomError("custom error occurred")


@action
async def handle_value_error() -> str:
    return "caught:ValueError"


@action
async def handle_type_error() -> str:
    return "caught:TypeError"


@action
async def handle_custom_error() -> str:
    return "caught:CustomError"


@workflow
class ExceptionMultiValueErrorWorkflow(Workflow):
    """Test workflow that raises ValueError and catches it with specific handler."""
    async def run(self):
        try:
            await raise_value_error()
        except ValueError:
            result = await handle_value_error()
        except TypeError:
            result = await handle_type_error()
        return result


@workflow
class ExceptionMultiTypeErrorWorkflow(Workflow):
    """Test workflow that raises TypeError and catches it with specific handler."""
    async def run(self):
        try:
            await raise_type_error()
        except ValueError:
            result = await handle_value_error()
        except TypeError:
            result = await handle_type_error()
        return result
