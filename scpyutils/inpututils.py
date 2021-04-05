"""
Utilities related to getting and validating user input.

Author: Sam Cohan
"""
import re
from typing import Any, Callable, Iterable, Optional, Union


def get_input(  # noqa: C901
    prompt: str,
    validator: Union[Callable[[Any], bool], Iterable, str],
    forced_type: Optional[type] = None,
    validation_failed_msg: Optional[str] = None,
    quit_keyword: str = "quit()",
) -> Any:
    """Get user input with typing and validation

    Args:
        prompt: prompt string so user know what input is expected.
        validator: If a Callable, the (optionally typed) input would be passed to it
            and would expect boolean output of whether the input is valid. If
            Iterable, then will check if input exists in the collection. If string,
            then it would be used as a regular expression to which the input must
            conform to.
        forced_type: if provided, would pass input to it to force the type. (e.g. int
            to force input to be integer) (default: None)
        validation_failed_msg: Override the default message when validation fails.
            (default: None and will have a reasonable message depending on validator
            type)
        quit_keyword: keyword to use to quit the input process (defaults to 'quit()')

    Returns:
        The validated result.
    """
    assert isinstance(
        validator, (Callable, Iterable, str)
    ), f"ERROR: validator={validator} must be Callable or Iterable"

    if validation_failed_msg is None:
        if isinstance(validator, Iterable):
            if isinstance(validator, str):
                validation_failed_msg = (
                    f"Please enter input matching regex='{validator}'."
                )
            elif isinstance(validator, Iterable):
                validation_failed_msg = f"Please enter input from {list(validator)}"
            else:
                validation_failed_msg = (
                    f"Please enter input which {validator} would accept!"
                )

    def force_type(x):
        if forced_type is None:
            return x
        if x == quit_keyword:
            return x
        try:
            return forced_type(x.strip())
        except:  # noqa: E722
            print("error forcing type")
        return None

    def validate(x):
        if isinstance(validator, Iterable):
            return x in validator
        if isinstance(validator, callable):
            return validator(x)
        if isinstance(validator, str):
            return re.match(validator, x)

    if not prompt.endswith(" "):
        prompt += " "

    res = force_type(input(prompt))
    while not validate(res):
        if res == quit_keyword:
            print("Aborting input!")
            return None
        print(f"input={res} is not valid. {validation_failed_msg}")
        res = force_type(input(prompt))

    return res
