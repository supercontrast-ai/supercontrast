from typing import Any

from supercontrast import Task, TaskMetadata


def print_request_response_and_metadata(
    task: Task,
    request: Any,
    response: Any,
    metadata: TaskMetadata,
):
    print("\n", "-" * 80, "\n")
    print(f"{task.value.capitalize()} Request:")
    print(request, "\n")
    print(f"{task.value.capitalize()} Response from {metadata.provider}:")
    print(response, "\n")
    print("Metadata:")
    print(metadata, "\n")
    print("-" * 80, "\n")
