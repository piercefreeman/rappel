# Architecture

Within our workers, every bit of Python logic that's run is technically an action. These can be explicit actions which are defined via an `@action` decorator or these can be implicit actions which are the bits of `Workflow.run` control flow that actually require a python interpreter to run.

Python Client Lib -> Rust Client Bridge -> DB
DB -> Rust Workflow Runner -> Python action runners

All Python library code is within one library - and all rust code is within one library as well. We compile the rust down to a maturin embedded executable for the rust client bridge - and a separate launchable bin entrypoint for the rust worker controller.

## python: client library

Parse what users intend to run via their Workflow instances, send the DAG definition to the database for execution. This python client library talks to the client bridge through embedded FII.

## rust: client bridge

While the Python client library could be in charge of upserting to the database by itself, all our other database management logic is stored in rust. So it makes more sense architecturally for it to manage both the client and worker db interactions. This also allows us to more easily support different runner languages in the future.

Takes care of versioning the workflow instance implementations. If the logic has changed we will automatically create a new version of the workflow instance - users will have to manually migrate old ones to the new version otherwise they will continue attempting to run with the older flow.

## rust: workflow runner

Launch python interpreters in parallel, by default 1 per core so we can work around the blocking limitations in the process bound GIL. The rust workflow runner communicates to the python subprocesses with structured messages sent over stdin/stdout.

## python: worker runners

Executing each actions within a python runtime environment. Handles importing the necessary first/third party dependencies with importlib.
