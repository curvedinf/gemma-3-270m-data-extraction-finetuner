"""
Fabric entrypoint for the Gemma 3 270M data-extraction fine-tuning pipeline.

Tasks are organized in collections (env, dataset, train, eval, package, ops)
to mirror the sections defined in PLAN.md. Each task delegates to helper
functions in `fab_tasks/` so logic stays modular and testable.
"""

from fabric import Collection

from fab_tasks import dataset, env, evaluate, ops, package, train


def build_namespace() -> Collection:
    """Create the root Fabric namespace with grouped sub-collections."""
    ns = Collection()

    env_ns = Collection("env")
    env_ns.add_task(env.bootstrap, "bootstrap")
    env_ns.add_task(env.lock, "lock")
    env_ns.add_task(env.check, "check")
    ns.add_collection(env_ns)

    dataset_ns = Collection("dataset")
    dataset_ns.add_task(dataset.pull, "pull")
    dataset_ns.add_task(dataset.clean, "clean")
    dataset_ns.add_task(dataset.split, "split")
    dataset_ns.add_task(dataset.stats, "stats")
    ns.add_collection(dataset_ns)

    train_ns = Collection("train")
    train_ns.add_task(train.prepare, "prepare")
    train_ns.add_task(train.run, "run")
    train_ns.add_task(train.resume, "resume")
    ns.add_collection(train_ns)

    eval_ns = Collection("eval")
    eval_ns.add_task(evaluate.generate, "generate")
    eval_ns.add_task(evaluate.judge, "judge")
    eval_ns.add_task(evaluate.report, "report")
    ns.add_collection(eval_ns)

    package_ns = Collection("package")
    package_ns.add_task(package.export, "export")
    ns.add_collection(package_ns)

    ops_ns = Collection("ops")
    ops_ns.add_task(ops.project_tokens, "project_tokens")
    ops_ns.add_task(ops.backfill, "backfill")
    ns.add_collection(ops_ns)

    return ns


ns = build_namespace()
