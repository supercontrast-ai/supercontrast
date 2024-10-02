import boto3
import logging
import threading
import time

from datetime import UTC, datetime, timedelta
from functools import wraps
from typing import Any, Callable

from supercontrast.provider import Provider
from supercontrast.task import Task


def track_cost(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args, **kwargs):
        if args and isinstance(args[0], Task) and isinstance(args[1], Provider):
            task = args[0]
            provider = args[1]
            if provider == Provider.AWS:
                start_time = datetime.now(UTC)
                last_month_start = start_time - timedelta(days=30)
                last_month_end = start_time

                # Get initial cost for the last month
                initial_cost = track_aws_cost(task, last_month_start, last_month_end)
                logging.info(f"Initial cost for the last month: ${initial_cost:.6f}")
                result = func(*args, **kwargs)

                # Set up hourly cost tracking
                def track_hourly_cost():
                    while True:
                        time.sleep(3600)  # Wait for an hour
                        current_time = datetime.now(UTC)
                        hourly_cost = track_aws_cost(task, start_time, current_time)
                        logging.info(
                            f"Cost incurred in the last hour: ${hourly_cost:.6f}"
                        )

                # Start hourly cost tracking in a separate thread
                cost_thread = threading.Thread(target=track_hourly_cost, daemon=True)
                cost_thread.start()

                return result
        return func(*args, **kwargs)

    return wrapper


def track_aws_cost(task: Task, start_time: datetime, end_time: datetime):
    if task == Task.OCR:
        service_name = "AmazonTextract"
    elif task == Task.TRANSLATION:
        service_name = "AmazonTranslate"
    elif task == Task.SENTIMENT_ANALYSIS:
        service_name = "AmazonComprehend"
    else:
        raise ValueError(f"Unsupported task: {task}")

    # Create a Cost Explorer client
    ce_client = boto3.client("ce", region_name="us-east-1")

    # Get cost and usage data
    response = ce_client.get_cost_and_usage(
        TimePeriod={"Start": start_time.isoformat(), "End": end_time.isoformat()},
        Granularity="HOURLY",
        Metrics=["UnblendedCost", "UsageQuantity"],
        GroupBy=[
            {"Type": "DIMENSION", "Key": "SERVICE"},
        ],
        Filter={"Dimensions": {"Key": "SERVICE", "Values": []}},
    )

    # Process the results
    for result in response["ResultsByTime"]:
        for group in result["Groups"]:
            cost = float(group["Metrics"]["UnblendedCost"]["Amount"])
            usage = float(group["Metrics"]["UsageQuantity"]["Amount"])

            if usage > 0:
                avg_cost_per_request = cost / usage
                logging.info(
                    f"Average cost per request for {task} in {service_name}: ${avg_cost_per_request:.6f}"
                )
            else:
                logging.info(
                    f"No usage recorded for {task} in {service_name} during this period."
                )

    # If no results, log a message
    if not response["ResultsByTime"]:
        logging.info(
            f"No cost data available for {task} in {service_name} during this period."
        )
