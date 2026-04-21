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


def upload_and_wait_for_processing(openai_client, path, label, timeout_seconds=600, poll_interval_seconds=5):
    """Upload a file and wait until it is processed before proceeding."""
    with open(path, "rb") as file_handle:
        uploaded = openai_client.files.create(file=file_handle, purpose="fine-tune")

    file_id = getattr(uploaded, "id", None)
    if not file_id:
        raise RuntimeError(f"{label} upload did not return a file id.")

    print(f"Uploaded {label} file: {file_id}. Waiting for processing...")

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        current = openai_client.files.retrieve(file_id)
        status = (getattr(current, "status", "") or "").lower()

        if status == "processed":
            print(f"{label} file processed successfully: {file_id}")
            return file_id

        if status in {"failed", "error", "cancelled"}:
            raise RuntimeError(
                f"{label} file processing failed for {file_id} with status '{status}'."
            )

        time.sleep(poll_interval_seconds)

    raise TimeoutError(
        f"Timed out after {timeout_seconds}s waiting for {label} file {file_id} to process."
    )


def submit_sft(args):
    credential = DefaultAzureCredential()

    project_endpoint = args.project_endpoint
    project_client = AIProjectClient(endpoint=project_endpoint, credential=credential)

    openai_client = project_client.get_openai_client() 

    train_file_id = upload_and_wait_for_processing(
        openai_client,
        args.training_file,
        "training",
    )
    val_file_id = upload_and_wait_for_processing(
        openai_client,
        args.validation_file,
        "validation",
    )

    hyperparameters = {}
    if args.epochs is not None:
        hyperparameters["n_epochs"] = args.epochs
    if args.batch_size is not None:
        hyperparameters["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        hyperparameters["learning_rate_multiplier"] = args.learning_rate

    supervised = {}
    if hyperparameters:
        supervised["hyperparameters"] = hyperparameters

    job_kwargs = dict(
        model=args.model,
        training_file=train_file_id,
        validation_file=val_file_id,
        method={
            "type": "supervised",
            "supervised": supervised,
        },
        suffix=args.suffix,
    )

    job = openai_client.fine_tuning.jobs.create(**job_kwargs)

    print("Submitted finetuning job:", job.id)


def build_parser():
    parser = HelpOnErrorParser(
        description="Submit SFT fine-tuning job",
        epilog=(
            "Example:\n"
            "  ./submit_sft.py --project-endpoint https://<resource>.services.ai.azure.com/api/projects/<project> --training-file train.jsonl --validation-file valid.jsonl --model gpt-4o-mini-2024-07-18"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--project-endpoint",
        required=True,
        help="Azure AI Project endpoint URL",
    )
    parser.add_argument("--training-file", required=True, help="Path to training JSONL file")
    parser.add_argument("--validation-file", required=True, help="Path to validation JSONL file")
    parser.add_argument("--model", required=True, help="Base model name")
    parser.add_argument("--suffix", default="sft-finetuned", help="Model name suffix")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate multiplier")
    return parser


if __name__ == "__main__":
    parser = build_parser()

    # If no arguments are provided, show help instead of submitting with defaults.
    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit(0)

    submit_sft(parser.parse_args())
