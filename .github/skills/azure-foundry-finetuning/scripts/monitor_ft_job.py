#!/usr/bin/env python3
# /// script
# dependencies = [
#   "azure-identity",
#   "azure-ai-projects",
# ]
# ///
import argparse
import sys
import time
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from common import HelpOnErrorParser


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


def monitor_ft_job(args):
	credential = DefaultAzureCredential()
	project_client = AIProjectClient(endpoint=args.project_endpoint, credential=credential)
	openai_client = project_client.get_openai_client()

	print(f"Monitoring fine-tuning job: {args.job_id}")
	while True:
		job = openai_client.fine_tuning.jobs.retrieve(args.job_id)
		status = (getattr(job, "status", "") or "").lower()
		print(f"Job status: {status}")

		if status in TERMINAL_STATUSES:
			print(f"Job reached terminal status: {status}")
			if status == "succeeded":
				return
			raise RuntimeError(f"Fine-tuning job ended with terminal status: {status}")

		time.sleep(args.poll_interval)


def build_parser():
	parser = HelpOnErrorParser(
		description="Monitor an SFT fine-tuning job until completion",
		epilog=(
			"Example:\n"
			"  ./monitor_ft_job.py --project-endpoint https://<resource>.services.ai.azure.com/api/projects/<project> --job-id ftjob_123"
		),
		formatter_class=argparse.RawTextHelpFormatter,
	)
	parser.add_argument(
		"--project-endpoint",
		required=True,
		help="Azure AI Project endpoint URL",
	)
	parser.add_argument("--job-id", required=True, help="Fine-tuning job ID")
	parser.add_argument(
		"--poll-interval",
		type=int,
		default=10,
		help="Polling interval in seconds (default: 10)",
	)
	return parser


if __name__ == "__main__":
	parser = build_parser()

	# If no arguments are provided, show help instead of failing with usage only.
	if len(sys.argv) == 1:
		parser.print_help()
		parser.exit(0)

	monitor_ft_job(parser.parse_args())
