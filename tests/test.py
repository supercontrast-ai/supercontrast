import boto3
import json

from datetime import datetime, timedelta


def get_cost_and_usage(service_codes, start_time, end_time):
    # Create a Cost Explorer client
    client = boto3.client("ce", region_name="us-east-1")

    # Get cost and usage data
    response = client.get_cost_and_usage(
        TimePeriod={"Start": start_time.isoformat(), "End": end_time.isoformat()},
        Granularity="HOURLY",
        Metrics=["UnblendedCost", "UsageQuantity"],
        GroupBy=[
            {"Type": "DIMENSION", "Key": "SERVICE"},
        ],
        Filter={"Dimensions": {"Key": "SERVICE", "Values": service_codes}},
    )

    return response


# Example usage
service_codes = ["Amazon Textract", "Amazon Rekognition", "Amazon Comprehend"]
end = datetime.utcnow()
start = end - timedelta(hours=24)  # Get data for the last 24 hours
cost_data = get_cost_and_usage(service_codes, start, end)

print(json.dumps(cost_data, indent=2))

# Process and print the results
for result in cost_data["ResultsByTime"]:
    print(
        f"\nTime period: {result['TimePeriod']['Start']} to {result['TimePeriod']['End']}"
    )
    for group in result["Groups"]:
        service = group["Keys"][0]
        cost = float(group["Metrics"]["UnblendedCost"]["Amount"])
        usage = float(group["Metrics"]["UsageQuantity"]["Amount"])
        print(f"  Service: {service}")
        print(f"    Cost: ${cost:.2f}")
        print(f"    Usage: {usage:.2f} {group['Metrics']['UsageQuantity']['Unit']}")
