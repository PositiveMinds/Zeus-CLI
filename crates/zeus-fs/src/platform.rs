//! Cloud / platform CLI integrations: GitHub (`gh`), Supabase, Vercel,
//! Netlify, Firebase, Fly.io, Railway, Render, AWS, Azure, Google Cloud,
//! Docker, Kubernetes (`kubectl`), Helm, Terraform, SAM/CloudFormation, and
//! CircleCI.
//!
//! Mirrors `git.rs` exactly: every operation shells out to the platform's
//! official CLI binary (keeping tokens/auth out of the agent — the user logs
//! in with `gh auth login`, `supabase login`, `vercel login`, etc.), and
//! every command routes through the same `PermissionGate`.
//!
//! Tiers (same shape as git):
//! - Read-only (list/view/status/logs/whoami/plan/diff) — allow.
//! - Mutating local or shared/network state (create/deploy/push/apply/
//!   destroy/run/link/close/merge/delete) — ask.

use crate::error::{FsError, Result};
use crate::permission::{ApprovalDecision, PermissionGate, PermissionRequest};
use std::path::PathBuf;
use std::process::Command;

#[derive(Debug, Clone, Default)]
pub struct PlatformOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub success: bool,
}

pub struct PlatformEngine {
    project_root: PathBuf,
    gate: PermissionGate,
}

impl PlatformEngine {
    pub fn new(project_root: PathBuf, gate: PermissionGate) -> Self {
        Self { project_root, gate }
    }

    fn run_bin(&self, bin: &str, args: &[String]) -> Result<PlatformOutput> {
        let output = Command::new(bin)
            .args(args)
            .current_dir(&self.project_root)
            .output()
            .map_err(|e| FsError::Other(format!("{bin} {args:?} failed to spawn: {e}")))?;
        Ok(PlatformOutput {
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            exit_code: output.status.code(),
            success: output.status.success(),
        })
    }

    fn enforce<F>(
        &self,
        tool: &str,
        description: String,
        command: String,
        approver: &mut F,
    ) -> Result<()>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.gate.enforce(
            &PermissionRequest {
                tool: tool.to_string(),
                path: None,
                command: Some(command),
                description,
                ..Default::default()
            },
            approver,
        )
    }

    fn enforce_strict(&self, tool: &str, description: String) -> Result<()> {
        self.gate.enforce_strict(&PermissionRequest {
            tool: tool.to_string(),
            path: None,
            command: None,
            description,
            ..Default::default()
        })
    }

    /// Check whether `bin` exists on PATH and its `--version` succeeds.
    fn available(&self, bin: &str) -> bool {
        Command::new(bin)
            .arg("--version")
            .current_dir(&self.project_root)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    fn require_bin(&self, bin: &str, op: &str, install_hint: &str) -> Result<()> {
        if !self.available(bin) {
            return Err(FsError::Other(format!(
                "{op} needs the `{bin}` CLI which is not on PATH. {install_hint}"
            )));
        }
        Ok(())
    }

    /// Run a read-only command with no user approval prompt.
    fn read(&self, tool: &str, description: &str, bin: &str, args: &[String]) -> Result<PlatformOutput> {
        self.enforce_strict(tool, description.to_string())?;
        self.run_bin(bin, args)
    }

    // ---------------------------------------------------------------
    // GitHub (`gh`)
    // ---------------------------------------------------------------

    pub fn gh_available(&self) -> bool {
        self.available("gh")
    }

    pub fn gh_issue_list(
        &self,
        state: &str,
        limit: usize,
        label: Option<&str>,
    ) -> Result<PlatformOutput> {
        self.require_bin("gh", "gh_issue_list", "Install it and run `gh auth login`.")?;
        let mut args = vec!["issue".into(), "list".into()];
        args.push("--state".into());
        args.push(state.to_string());
        if let Some(label) = label {
            args.push("--label".into());
            args.push(label.to_string());
        }
        args.push("--limit".into());
        args.push(limit.to_string());
        self.read("gh_issue_list", "gh issue list", "gh", &args)
    }

    pub fn gh_issue_view(&self, number: &str) -> Result<PlatformOutput> {
        self.require_bin("gh", "gh_issue_view", "Install it and run `gh auth login`.")?;
        let args = vec!["issue".into(), "view".into(), number.to_string()];
        self.read("gh_issue_view", "gh issue view", "gh", &args)
    }

    pub fn gh_issue_create<F>(
        &self,
        title: &str,
        body: Option<&str>,
        label: Option<&str>,
        approver: &mut F,
    ) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("gh", "gh_issue_create", "Install it and run `gh auth login`.")?;
        let command = "gh issue create";
        self.enforce(
            "gh_issue_create",
            "create a GitHub issue".into(),
            command.into(),
            approver,
        )?;
        let mut args = vec!["issue".into(), "create".into(), "--title".into(), title.to_string()];
        if let Some(body) = body {
            args.push("--body".into());
            args.push(body.to_string());
        }
        if let Some(label) = label {
            args.push("--label".into());
            args.push(label.to_string());
        }
        self.run_bin("gh", &args)
    }

    pub fn gh_issue_close<F>(&self, number: &str, approver: &mut F) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("gh", "gh_issue_close", "Install it and run `gh auth login`.")?;
        self.enforce(
            "gh_issue_close",
            "close a GitHub issue".into(),
            "gh issue close".into(),
            approver,
        )?;
        let args = vec!["issue".into(), "close".into(), number.to_string()];
        self.run_bin("gh", &args)
    }

    pub fn gh_pr_list(&self, state: &str, limit: usize) -> Result<PlatformOutput> {
        self.require_bin("gh", "gh_pr_list", "Install it and run `gh auth login`.")?;
        let mut args = vec!["pr".into(), "list".into(), "--state".into(), state.to_string()];
        args.push("--limit".into());
        args.push(limit.to_string());
        self.read("gh_pr_list", "gh pr list", "gh", &args)
    }

    pub fn gh_pr_view(&self, number: &str) -> Result<PlatformOutput> {
        self.require_bin("gh", "gh_pr_view", "Install it and run `gh auth login`.")?;
        let args = vec!["pr".into(), "view".into(), number.to_string()];
        self.read("gh_pr_view", "gh pr view", "gh", &args)
    }

    pub fn gh_pr_create<F>(
        &self,
        title: &str,
        body: Option<&str>,
        base: Option<&str>,
        approver: &mut F,
    ) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("gh", "gh_pr_create", "Install it and run `gh auth login`.")?;
        self.enforce(
            "gh_pr_create",
            "create a GitHub pull request".into(),
            "gh pr create".into(),
            approver,
        )?;
        let mut args = vec!["pr".into(), "create".into(), "--title".into(), title.to_string()];
        if let Some(body) = body {
            args.push("--body".into());
            args.push(body.to_string());
        }
        if let Some(base) = base {
            args.push("--base".into());
            args.push(base.to_string());
        }
        self.run_bin("gh", &args)
    }

    pub fn gh_pr_merge<F>(
        &self,
        number: &str,
        method: Option<&str>,
        delete_branch: bool,
        approver: &mut F,
    ) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("gh", "gh_pr_merge", "Install it and run `gh auth login`.")?;
        self.enforce(
            "gh_pr_merge",
            "merge a GitHub pull request".into(),
            "gh pr merge".into(),
            approver,
        )?;
        let mut args = vec!["pr".into(), "merge".into(), number.to_string()];
        if let Some(method) = method {
            args.push(format!("--{method}"));
        }
        if delete_branch {
            args.push("--delete-branch".into());
        }
        self.run_bin("gh", &args)
    }

    pub fn gh_release_list(&self, limit: usize) -> Result<PlatformOutput> {
        self.require_bin("gh", "gh_release_list", "Install it and run `gh auth login`.")?;
        let mut args = vec!["release".into(), "list".into()];
        args.push("--limit".into());
        args.push(limit.to_string());
        self.read("gh_release_list", "gh release list", "gh", &args)
    }

    pub fn gh_release_create<F>(
        &self,
        tag: &str,
        title: Option<&str>,
        notes: Option<&str>,
        approver: &mut F,
    ) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("gh", "gh_release_create", "Install it and run `gh auth login`.")?;
        self.enforce(
            "gh_release_create",
            "create a GitHub release".into(),
            "gh release create".into(),
            approver,
        )?;
        let mut args = vec!["release".into(), "create".into(), tag.to_string()];
        if let Some(title) = title {
            args.push("--title".into());
            args.push(title.to_string());
        }
        if let Some(notes) = notes {
            args.push("--notes".into());
            args.push(notes.to_string());
        }
        self.run_bin("gh", &args)
    }

    pub fn gh_workflow_list(&self) -> Result<PlatformOutput> {
        self.require_bin("gh", "gh_workflow_list", "Install it and run `gh auth login`.")?;
        let args = vec!["workflow".into(), "list".into()];
        self.read("gh_workflow_list", "gh workflow list", "gh", &args)
    }

    pub fn gh_workflow_run<F>(
        &self,
        workflow: &str,
        ref_: Option<&str>,
        approver: &mut F,
    ) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("gh", "gh_workflow_run", "Install it and run `gh auth login`.")?;
        self.enforce(
            "gh_workflow_run",
            "trigger a GitHub Actions workflow".into(),
            "gh workflow run".into(),
            approver,
        )?;
        let mut args = vec!["workflow".into(), "run".into(), workflow.to_string()];
        if let Some(ref_) = ref_ {
            args.push("--ref".into());
            args.push(ref_.to_string());
        }
        self.run_bin("gh", &args)
    }

    pub fn gh_run_list(&self, workflow: Option<&str>, limit: usize) -> Result<PlatformOutput> {
        self.require_bin("gh", "gh_run_list", "Install it and run `gh auth login`.")?;
        let mut args = vec!["run".into(), "list".into()];
        if let Some(workflow) = workflow {
            args.push("--workflow".into());
            args.push(workflow.to_string());
        }
        args.push("--limit".into());
        args.push(limit.to_string());
        self.read("gh_run_list", "gh run list", "gh", &args)
    }

    // ---------------------------------------------------------------
    // Supabase
    // ---------------------------------------------------------------

    pub fn supabase_available(&self) -> bool {
        self.available("supabase")
    }

    pub fn supabase_login<F>(&self, approver: &mut F) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("supabase", "supabase_login", "Install it (`brew install supabase/tap/supabase`).")?;
        self.enforce(
            "supabase_login",
            "log in to Supabase (opens browser)".into(),
            "supabase login".into(),
            approver,
        )?;
        let args = vec!["login".into()];
        self.run_bin("supabase", &args)
    }

    pub fn supabase_link<F>(&self, project_ref: Option<&str>, approver: &mut F) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("supabase", "supabase_link", "Install it (`brew install supabase/tap/supabase`).")?;
        self.enforce(
            "supabase_link",
            "link the project to a Supabase remote".into(),
            "supabase link".into(),
            approver,
        )?;
        let mut args = vec!["link".into()];
        if let Some(project_ref) = project_ref {
            args.push("--project-ref".into());
            args.push(project_ref.to_string());
        }
        self.run_bin("supabase", &args)
    }

    pub fn supabase_projects_list(&self) -> Result<PlatformOutput> {
        self.require_bin("supabase", "supabase_projects_list", "Install it (`brew install supabase/tap/supabase`).")?;
        let args = vec!["projects".into(), "list".into()];
        self.read("supabase_projects_list", "supabase projects list", "supabase", &args)
    }

    pub fn supabase_status(&self) -> Result<PlatformOutput> {
        self.require_bin("supabase", "supabase_status", "Install it (`brew install supabase/tap/supabase`).")?;
        let args = vec!["status".into()];
        self.read("supabase_status", "supabase status (local dev services)", "supabase", &args)
    }

    pub fn supabase_db_push<F>(&self, approver: &mut F) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("supabase", "supabase_db_push", "Install it (`brew install supabase/tap/supabase`).")?;
        self.enforce(
            "supabase_db_push",
            "push local migrations to the linked remote database".into(),
            "supabase db push".into(),
            approver,
        )?;
        let args = vec!["db".into(), "push".into()];
        self.run_bin("supabase", &args)
    }

    pub fn supabase_db_diff(&self, schema: Option<&str>, linked: bool) -> Result<PlatformOutput> {
        self.require_bin("supabase", "supabase_db_diff", "Install it (`brew install supabase/tap/supabase`).")?;
        let mut args = vec!["db".into(), "diff".into()];
        if linked {
            args.push("--linked".into());
        }
        if let Some(schema) = schema {
            args.push("--schema".into());
            args.push(schema.to_string());
        }
        self.read("supabase_db_diff", "supabase db diff", "supabase", &args)
    }

    pub fn supabase_functions_list(&self) -> Result<PlatformOutput> {
        self.require_bin("supabase", "supabase_functions_list", "Install it (`brew install supabase/tap/supabase`).")?;
        let args = vec!["functions".into(), "list".into()];
        self.read("supabase_functions_list", "supabase functions list", "supabase", &args)
    }

    pub fn supabase_functions_deploy<F>(
        &self,
        function: &str,
        project_ref: Option<&str>,
        no_verify_jwt: bool,
        approver: &mut F,
    ) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("supabase", "supabase_functions_deploy", "Install it (`brew install supabase/tap/supabase`).")?;
        self.enforce(
            "supabase_functions_deploy",
            "deploy a Supabase Edge Function".into(),
            "supabase functions deploy".into(),
            approver,
        )?;
        let mut args = vec!["functions".into(), "deploy".into(), function.to_string()];
        if let Some(project_ref) = project_ref {
            args.push("--project-ref".into());
            args.push(project_ref.to_string());
        }
        if no_verify_jwt {
            args.push("--no-verify-jwt".into());
        }
        self.run_bin("supabase", &args)
    }

    // ---------------------------------------------------------------
    // Vercel
    // ---------------------------------------------------------------

    pub fn vercel_available(&self) -> bool {
        self.available("vercel")
    }

    pub fn vercel_whoami(&self) -> Result<PlatformOutput> {
        self.require_bin("vercel", "vercel_whoami", "Install it (`npm i -g vercel`) and run `vercel login`.")?;
        let args = vec!["whoami".into()];
        self.read("vercel_whoami", "vercel whoami", "vercel", &args)
    }

    pub fn vercel_projects_list(&self) -> Result<PlatformOutput> {
        self.require_bin("vercel", "vercel_projects_list", "Install it (`npm i -g vercel`) and run `vercel login`.")?;
        let args = vec!["projects".into(), "ls".into()];
        self.read("vercel_projects_list", "vercel projects ls", "vercel", &args)
    }

    pub fn vercel_env_list(
        &self,
        env: Option<&str>,
        project: Option<&str>,
    ) -> Result<PlatformOutput> {
        self.require_bin("vercel", "vercel_env_list", "Install it (`npm i -g vercel`) and run `vercel login`.")?;
        let mut args = vec!["env".into(), "ls".into()];
        if let Some(env) = env {
            args.push(env.to_string());
        }
        if let Some(project) = project {
            args.push("--project".into());
            args.push(project.to_string());
        }
        self.read("vercel_env_list", "vercel env ls", "vercel", &args)
    }

    pub fn vercel_deploy<F>(
        &self,
        prod: bool,
        target: Option<&str>,
        project: Option<&str>,
        approver: &mut F,
    ) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("vercel", "vercel_deploy", "Install it (`npm i -g vercel`) and run `vercel login`.")?;
        self.enforce(
            "vercel_deploy",
            "deploy to Vercel".into(),
            "vercel deploy".into(),
            approver,
        )?;
        let mut args = vec!["deploy".into()];
        if prod {
            args.push("--prod".into());
        }
        if let Some(target) = target {
            args.push(target.to_string());
        }
        if let Some(project) = project {
            args.push("--project".into());
            args.push(project.to_string());
        }
        self.run_bin("vercel", &args)
    }

    pub fn vercel_logs(
        &self,
        deployment: Option<&str>,
        project: Option<&str>,
        follow: bool,
    ) -> Result<PlatformOutput> {
        self.require_bin("vercel", "vercel_logs", "Install it (`npm i -g vercel`) and run `vercel login`.")?;
        let mut args = vec!["logs".into()];
        if let Some(deployment) = deployment {
            args.push(deployment.to_string());
        }
        if follow {
            args.push("--follow".into());
        }
        if let Some(project) = project {
            args.push("--project".into());
            args.push(project.to_string());
        }
        self.read("vercel_logs", "vercel logs", "vercel", &args)
    }

    // ---------------------------------------------------------------
    // Docker
    // ---------------------------------------------------------------

    pub fn docker_available(&self) -> bool {
        self.available("docker")
    }

    pub fn docker_ps(&self, all: bool) -> Result<PlatformOutput> {
        self.require_bin("docker", "docker_ps", "Install Docker Desktop / the docker CLI.")?;
        let mut args = vec!["ps".into()];
        if all {
            args.push("--all".into());
        }
        self.read("docker_ps", "docker ps", "docker", &args)
    }

    pub fn docker_images(&self) -> Result<PlatformOutput> {
        self.require_bin("docker", "docker_images", "Install Docker Desktop / the docker CLI.")?;
        let args = vec!["images".into()];
        self.read("docker_images", "docker images", "docker", &args)
    }

    pub fn docker_compose_up<F>(
        &self,
        services: Vec<String>,
        detached: bool,
        build: bool,
        approver: &mut F,
    ) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("docker", "docker_compose_up", "Install Docker Desktop / the docker CLI.")?;
        self.enforce(
            "docker_compose_up",
            "docker compose up".into(),
            "docker compose up".into(),
            approver,
        )?;
        let mut args = vec!["compose".into(), "up".into()];
        if detached {
            args.push("-d".into());
        }
        if build {
            args.push("--build".into());
        }
        args.extend(services);
        self.run_bin("docker", &args)
    }

    pub fn docker_compose_down<F>(&self, volumes: bool, approver: &mut F) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("docker", "docker_compose_down", "Install Docker Desktop / the docker CLI.")?;
        self.enforce(
            "docker_compose_down",
            "docker compose down".into(),
            "docker compose down".into(),
            approver,
        )?;
        let mut args = vec!["compose".into(), "down".into()];
        if volumes {
            args.push("-v".into());
        }
        self.run_bin("docker", &args)
    }

    pub fn docker_compose_logs(
        &self,
        service: Option<&str>,
        follow: bool,
    ) -> Result<PlatformOutput> {
        self.require_bin("docker", "docker_compose_logs", "Install Docker Desktop / the docker CLI.")?;
        let mut args = vec!["compose".into(), "logs".into()];
        if follow {
            args.push("--follow".into());
        }
        if let Some(service) = service {
            args.push(service.to_string());
        }
        self.read("docker_compose_logs", "docker compose logs", "docker", &args)
    }

    // ---------------------------------------------------------------
    // Kubernetes (`kubectl`)
    // ---------------------------------------------------------------

    pub fn kubectl_available(&self) -> bool {
        self.available("kubectl")
    }

    pub fn k8s_get(
        &self,
        resource: &str,
        name: Option<&str>,
        namespace: Option<&str>,
        all_namespaces: bool,
    ) -> Result<PlatformOutput> {
        self.require_bin("kubectl", "k8s_get", "Install kubectl and configure a kubeconfig.")?;
        let mut args = vec!["get".into(), resource.to_string()];
        if let Some(name) = name {
            args.push(name.to_string());
        }
        if let Some(namespace) = namespace {
            args.push("-n".into());
            args.push(namespace.to_string());
        }
        if all_namespaces {
            args.push("--all-namespaces".into());
        }
        self.read("k8s_get", "kubectl get", "kubectl", &args)
    }

    pub fn k8s_logs(
        &self,
        pod: &str,
        container: Option<&str>,
        namespace: Option<&str>,
        follow: bool,
    ) -> Result<PlatformOutput> {
        self.require_bin("kubectl", "k8s_logs", "Install kubectl and configure a kubeconfig.")?;
        let mut args = vec!["logs".into(), pod.to_string()];
        if let Some(container) = container {
            args.push("-c".into());
            args.push(container.to_string());
        }
        if let Some(namespace) = namespace {
            args.push("-n".into());
            args.push(namespace.to_string());
        }
        if follow {
            args.push("--follow".into());
        }
        self.read("k8s_logs", "kubectl logs", "kubectl", &args)
    }

    pub fn k8s_apply<F>(
        &self,
        path: &str,
        namespace: Option<&str>,
        approver: &mut F,
    ) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("kubectl", "k8s_apply", "Install kubectl and configure a kubeconfig.")?;
        self.enforce(
            "k8s_apply",
            "kubectl apply".into(),
            "kubectl apply -f".into(),
            approver,
        )?;
        let mut args = vec!["apply".into(), "-f".into(), path.to_string()];
        if let Some(namespace) = namespace {
            args.push("-n".into());
            args.push(namespace.to_string());
        }
        self.run_bin("kubectl", &args)
    }

    pub fn k8s_rollout_status(
        &self,
        resource: &str,
        namespace: Option<&str>,
    ) -> Result<PlatformOutput> {
        self.require_bin("kubectl", "k8s_rollout_status", "Install kubectl and configure a kubeconfig.")?;
        let mut args = vec!["rollout".into(), "status".into(), resource.to_string()];
        if let Some(namespace) = namespace {
            args.push("-n".into());
            args.push(namespace.to_string());
        }
        self.read("k8s_rollout_status", "kubectl rollout status", "kubectl", &args)
    }

    // ---------------------------------------------------------------
    // Terraform
    // ---------------------------------------------------------------

    pub fn terraform_available(&self) -> bool {
        self.available("terraform")
    }

    pub fn tf_init<F>(&self, approver: &mut F) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("terraform", "tf_init", "Install terraform / OpenTofu.")?;
        self.enforce("tf_init", "terraform init".into(), "terraform init".into(), approver)?;
        let args = vec!["init".into()];
        self.run_bin("terraform", &args)
    }

    pub fn tf_validate(&self) -> Result<PlatformOutput> {
        self.require_bin("terraform", "tf_validate", "Install terraform / OpenTofu.")?;
        let args = vec!["validate".into()];
        self.read("tf_validate", "terraform validate", "terraform", &args)
    }

    pub fn tf_plan(&self, out: Option<&str>) -> Result<PlatformOutput> {
        self.require_bin("terraform", "tf_plan", "Install terraform / OpenTofu.")?;
        let mut args = vec!["plan".into()];
        if let Some(out) = out {
            args.push(format!("-out={out}"));
        }
        self.read("tf_plan", "terraform plan", "terraform", &args)
    }

    pub fn tf_apply<F>(&self, plan_file: Option<&str>, auto_approve: bool, approver: &mut F) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("terraform", "tf_apply", "Install terraform / OpenTofu.")?;
        self.enforce("tf_apply", "terraform apply".into(), "terraform apply".into(), approver)?;
        let mut args = vec!["apply".into()];
        if auto_approve {
            args.push("-auto-approve".into());
        }
        if let Some(plan_file) = plan_file {
            args.push(plan_file.to_string());
        }
        self.run_bin("terraform", &args)
    }

    // ---------------------------------------------------------------
    // CircleCI
    // ---------------------------------------------------------------

    pub fn circleci_available(&self) -> bool {
        self.available("circleci")
    }

    pub fn circleci_validate(&self, config: Option<&str>) -> Result<PlatformOutput> {
        self.require_bin("circleci", "circleci_validate", "Install it (`brew install circleci`) and run `circleci setup`.")?;
        let mut args = vec!["config".into(), "validate".into()];
        if let Some(config) = config {
            args.push(config.to_string());
        }
        self.read("circleci_validate", "circleci config validate", "circleci", &args)
    }

    pub fn circleci_builds(
        &self,
        project: &str,
        branch: Option<&str>,
        limit: usize,
    ) -> Result<PlatformOutput> {
        self.require_bin("circleci", "circleci_builds", "Install it (`brew install circleci`) and run `circleci setup`.")?;
        let mut args = vec!["builds".into(), project.to_string()];
        if let Some(branch) = branch {
            args.push(format!("--branch={branch}"));
        }
        args.push(format!("--limit={limit}"));
        self.read("circleci_builds", "circleci builds", "circleci", &args)
    }

    // ---------------------------------------------------------------
    // AWS (`aws`) — S3 / ECS / Lambda / ECR
    // ---------------------------------------------------------------

    pub fn aws_available(&self) -> bool {
        self.available("aws")
    }

    pub fn aws_whoami(&self) -> Result<PlatformOutput> {
        self.require_bin("aws", "aws_whoami", "Install AWS CLI v2 (`winget install Amazon.AWSCLI`) and configure credentials.")?;
        let args = vec!["sts".into(), "get-caller-identity".into()];
        self.read("aws_whoami", "aws sts get-caller-identity", "aws", &args)
    }

    pub fn aws_s3_ls(&self, path: Option<&str>) -> Result<PlatformOutput> {
        self.require_bin("aws", "aws_s3_ls", "Install AWS CLI v2 (`winget install Amazon.AWSCLI`).")?;
        let mut args = vec!["s3".into(), "ls".into()];
        if let Some(path) = path {
            args.push(path.to_string());
        }
        self.read("aws_s3_ls", "aws s3 ls", "aws", &args)
    }

    pub fn aws_s3_sync<F>(&self, source: &str, dest: &str, approver: &mut F) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("aws", "aws_s3_sync", "Install AWS CLI v2 (`winget install Amazon.AWSCLI`).")?;
        self.enforce(
            "aws_s3_sync",
            "sync files to/from an S3 bucket".into(),
            "aws s3 sync".into(),
            approver,
        )?;
        let args = vec!["s3".into(), "sync".into(), source.to_string(), dest.to_string()];
        self.run_bin("aws", &args)
    }

    pub fn aws_ecr_login<F>(&self, approver: &mut F) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("aws", "aws_ecr_login", "Install AWS CLI v2 (`winget install Amazon.AWSCLI`).")?;
        self.enforce(
            "aws_ecr_login",
            "log in to Amazon ECR (docker login)".into(),
            "aws ecr get-login-password | docker login".into(),
            approver,
        )?;
        let args = vec!["ecr".into(), "get-login-password".into()];
        self.run_bin("aws", &args)
    }

    pub fn aws_lambda_list(&self) -> Result<PlatformOutput> {
        self.require_bin("aws", "aws_lambda_list", "Install AWS CLI v2 (`winget install Amazon.AWSCLI`).")?;
        let args = vec!["lambda".into(), "list-functions".into()];
        self.read("aws_lambda_list", "aws lambda list-functions", "aws", &args)
    }

    pub fn aws_lambda_invoke<F>(&self, function: &str, payload: Option<&str>, approver: &mut F) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("aws", "aws_lambda_invoke", "Install AWS CLI v2 (`winget install Amazon.AWSCLI`).")?;
        self.enforce(
            "aws_lambda_invoke",
            "invoke an AWS Lambda function".into(),
            "aws lambda invoke".into(),
            approver,
        )?;
        let mut args = vec!["lambda".into(), "invoke".into(), "--function-name".into(), function.to_string()];
        if let Some(payload) = payload {
            args.push("--payload".into());
            args.push(payload.to_string());
        }
        args.push("out.json".into());
        self.run_bin("aws", &args)
    }

    pub fn aws_ecs_list_clusters(&self) -> Result<PlatformOutput> {
        self.require_bin("aws", "aws_ecs_list_clusters", "Install AWS CLI v2 (`winget install Amazon.AWSCLI`).")?;
        let args = vec!["ecs".into(), "list-clusters".into()];
        self.read("aws_ecs_list_clusters", "aws ecs list-clusters", "aws", &args)
    }

    pub fn aws_ecs_force_deploy<F>(&self, cluster: &str, service: &str, approver: &mut F) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("aws", "aws_ecs_force_deploy", "Install AWS CLI v2 (`winget install Amazon.AWSCLI`).")?;
        self.enforce(
            "aws_ecs_force_deploy",
            "force a new deployment of an ECS service".into(),
            "aws ecs update-service --force-new-deployment".into(),
            approver,
        )?;
        let args = vec![
            "ecs".into(),
            "update-service".into(),
            "--cluster".into(),
            cluster.to_string(),
            "--service".into(),
            service.to_string(),
            "--force-new-deployment".into(),
        ];
        self.run_bin("aws", &args)
    }

    // ---------------------------------------------------------------
    // AWS SAM / CloudFormation (`sam` / `aws cloudformation`)
    // ---------------------------------------------------------------

    pub fn sam_available(&self) -> bool {
        self.available("sam")
    }

    pub fn sam_build<F>(&self, approver: &mut F) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("sam", "sam_build", "Install AWS SAM CLI (`winget install AWS.SAMCLI`).")?;
        self.enforce("sam_build", "sam build".into(), "sam build".into(), approver)?;
        let args = vec!["build".into()];
        self.run_bin("sam", &args)
    }

    pub fn sam_deploy<F>(&self, guided: bool, stack_name: Option<&str>, approver: &mut F) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("sam", "sam_deploy", "Install AWS SAM CLI (`winget install AWS.SAMCLI`).")?;
        self.enforce(
            "sam_deploy",
            "deploy an AWS SAM stack".into(),
            "sam deploy".into(),
            approver,
        )?;
        let mut args = vec!["deploy".into()];
        if guided {
            args.push("--guided".into());
        }
        if let Some(stack_name) = stack_name {
            args.push("--stack-name".into());
            args.push(stack_name.to_string());
        }
        self.run_bin("sam", &args)
    }

    pub fn cloudformation_describe(&self, stack: &str) -> Result<PlatformOutput> {
        self.require_bin("aws", "cloudformation_describe", "Install AWS CLI v2 (`winget install Amazon.AWSCLI`).")?;
        let args = vec!["cloudformation".into(), "describe-stacks".into(), "--stack-name".into(), stack.to_string()];
        self.read("cloudformation_describe", "aws cloudformation describe-stacks", "aws", &args)
    }

    pub fn cloudformation_deploy<F>(&self, template: &str, stack: &str, approver: &mut F) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("aws", "cloudformation_deploy", "Install AWS CLI v2 (`winget install Amazon.AWSCLI`).")?;
        self.enforce(
            "cloudformation_deploy",
            "create/update a CloudFormation stack".into(),
            "aws cloudformation deploy".into(),
            approver,
        )?;
        let args = vec![
            "cloudformation".into(),
            "deploy".into(),
            "--template-file".into(),
            template.to_string(),
            "--stack-name".into(),
            stack.to_string(),
        ];
        self.run_bin("aws", &args)
    }

    // ---------------------------------------------------------------
    // Azure (`az`)
    // ---------------------------------------------------------------

    pub fn az_available(&self) -> bool {
        self.available("az")
    }

    pub fn az_whoami(&self) -> Result<PlatformOutput> {
        self.require_bin("az", "az_whoami", "Install Azure CLI (`winget install Microsoft.AzureCLI`).")?;
        let args = vec!["account".into(), "show".into()];
        self.read("az_whoami", "az account show", "az", &args)
    }

    pub fn az_webapp_list(&self) -> Result<PlatformOutput> {
        self.require_bin("az", "az_webapp_list", "Install Azure CLI (`winget install Microsoft.AzureCLI`).")?;
        let args = vec!["webapp".into(), "list".into()];
        self.read("az_webapp_list", "az webapp list", "az", &args)
    }

    pub fn az_webapp_deploy<F>(&self, name: &str, resource_group: &str, source: &str, approver: &mut F) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("az", "az_webapp_deploy", "Install Azure CLI (`winget install Microsoft.AzureCLI`).")?;
        self.enforce(
            "az_webapp_deploy",
            "deploy to an Azure App Service web app".into(),
            "az webapp deploy".into(),
            approver,
        )?;
        let args = vec![
            "webapp".into(),
            "deploy".into(),
            "--name".into(),
            name.to_string(),
            "--resource-group".into(),
            resource_group.to_string(),
            "--src-path".into(),
            source.to_string(),
        ];
        self.run_bin("az", &args)
    }

    pub fn az_functionapp_deploy<F>(&self, name: &str, resource_group: &str, source: &str, approver: &mut F) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("az", "az_functionapp_deploy", "Install Azure CLI (`winget install Microsoft.AzureCLI`).")?;
        self.enforce(
            "az_functionapp_deploy",
            "deploy an Azure Functions app".into(),
            "az functionapp deployment source config-zip".into(),
            approver,
        )?;
        let args = vec![
            "functionapp".into(),
            "deployment".into(),
            "source".into(),
            "config-zip".into(),
            "--name".into(),
            name.to_string(),
            "--resource-group".into(),
            resource_group.to_string(),
            "--src".into(),
            source.to_string(),
        ];
        self.run_bin("az", &args)
    }

    // ---------------------------------------------------------------
    // Google Cloud (`gcloud`)
    // ---------------------------------------------------------------

    pub fn gcloud_available(&self) -> bool {
        self.available("gcloud")
    }

    pub fn gcloud_whoami(&self) -> Result<PlatformOutput> {
        self.require_bin("gcloud", "gcloud_whoami", "Install Google Cloud SDK (https://cloud.google.com/sdk).")?;
        let args = vec!["config".into(), "list".into()];
        self.read("gcloud_whoami", "gcloud config list", "gcloud", &args)
    }

    pub fn gcloud_app_deploy<F>(&self, approver: &mut F) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("gcloud", "gcloud_app_deploy", "Install Google Cloud SDK (https://cloud.google.com/sdk).")?;
        self.enforce(
            "gcloud_app_deploy",
            "deploy to Google App Engine".into(),
            "gcloud app deploy".into(),
            approver,
        )?;
        let args = vec!["app".into(), "deploy".into()];
        self.run_bin("gcloud", &args)
    }

    pub fn gcloud_run_deploy<F>(&self, service: &str, image: &str, region: Option<&str>, approver: &mut F) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("gcloud", "gcloud_run_deploy", "Install Google Cloud SDK (https://cloud.google.com/sdk).")?;
        self.enforce(
            "gcloud_run_deploy",
            "deploy a container to Cloud Run".into(),
            "gcloud run deploy".into(),
            approver,
        )?;
        let mut args = vec![
            "run".into(),
            "deploy".into(),
            service.to_string(),
            "--image".into(),
            image.to_string(),
        ];
        if let Some(region) = region {
            args.push("--region".into());
            args.push(region.to_string());
        }
        args.push("--platform".into());
        args.push("managed".into());
        self.run_bin("gcloud", &args)
    }

    pub fn gcloud_run_services(&self) -> Result<PlatformOutput> {
        self.require_bin("gcloud", "gcloud_run_services", "Install Google Cloud SDK (https://cloud.google.com/sdk).")?;
        let args = vec!["run".into(), "services".into(), "list".into()];
        self.read("gcloud_run_services", "gcloud run services list", "gcloud", &args)
    }

    // ---------------------------------------------------------------
    // Helm
    // ---------------------------------------------------------------

    pub fn helm_available(&self) -> bool {
        self.available("helm")
    }

    pub fn helm_list(&self, namespace: Option<&str>) -> Result<PlatformOutput> {
        self.require_bin("helm", "helm_list", "Install Helm (https://helm.sh).")?;
        let mut args = vec!["list".into()];
        if let Some(namespace) = namespace {
            args.push("-n".into());
            args.push(namespace.to_string());
        }
        self.read("helm_list", "helm list", "helm", &args)
    }

    pub fn helm_status(&self, release: &str, namespace: Option<&str>) -> Result<PlatformOutput> {
        self.require_bin("helm", "helm_status", "Install Helm (https://helm.sh).")?;
        let mut args = vec!["status".into(), release.to_string()];
        if let Some(namespace) = namespace {
            args.push("-n".into());
            args.push(namespace.to_string());
        }
        self.read("helm_status", "helm status", "helm", &args)
    }

    pub fn helm_install<F>(&self, release: &str, chart: &str, namespace: Option<&str>, approver: &mut F) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("helm", "helm_install", "Install Helm (https://helm.sh).")?;
        self.enforce(
            "helm_install",
            "install a Helm chart".into(),
            "helm install".into(),
            approver,
        )?;
        let mut args = vec!["install".into(), release.to_string(), chart.to_string()];
        if let Some(namespace) = namespace {
            args.push("-n".into());
            args.push(namespace.to_string());
        }
        self.run_bin("helm", &args)
    }

    pub fn helm_upgrade<F>(&self, release: &str, chart: &str, namespace: Option<&str>, approver: &mut F) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("helm", "helm_upgrade", "Install Helm (https://helm.sh).")?;
        self.enforce(
            "helm_upgrade",
            "upgrade a Helm release".into(),
            "helm upgrade".into(),
            approver,
        )?;
        let mut args = vec!["upgrade".into(), release.to_string(), chart.to_string()];
        if let Some(namespace) = namespace {
            args.push("-n".into());
            args.push(namespace.to_string());
        }
        self.run_bin("helm", &args)
    }

    pub fn helm_uninstall<F>(&self, release: &str, namespace: Option<&str>, approver: &mut F) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("helm", "helm_uninstall", "Install Helm (https://helm.sh).")?;
        self.enforce(
            "helm_uninstall",
            "uninstall a Helm release".into(),
            "helm uninstall".into(),
            approver,
        )?;
        let mut args = vec!["uninstall".into(), release.to_string()];
        if let Some(namespace) = namespace {
            args.push("-n".into());
            args.push(namespace.to_string());
        }
        self.run_bin("helm", &args)
    }

    // ---------------------------------------------------------------
    // Fly.io (`flyctl`)
    // ---------------------------------------------------------------

    pub fn fly_available(&self) -> bool {
        self.available("flyctl")
    }

    pub fn fly_whoami(&self) -> Result<PlatformOutput> {
        self.require_bin("flyctl", "fly_whoami", "Install flyctl (`winget install fly-io.flyctl`).")?;
        let args = vec!["auth".into(), "whoami".into()];
        self.read("fly_whoami", "flyctl auth whoami", "flyctl", &args)
    }

    pub fn fly_apps_list(&self) -> Result<PlatformOutput> {
        self.require_bin("flyctl", "fly_apps_list", "Install flyctl (`winget install fly-io.flyctl`).")?;
        let args = vec!["apps".into(), "list".into()];
        self.read("fly_apps_list", "flyctl apps list", "flyctl", &args)
    }

    pub fn fly_deploy<F>(&self, image: Option<&str>, app: Option<&str>, approver: &mut F) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("flyctl", "fly_deploy", "Install flyctl (`winget install fly-io.flyctl`).")?;
        self.enforce(
            "fly_deploy",
            "deploy to Fly.io".into(),
            "flyctl deploy".into(),
            approver,
        )?;
        let mut args = vec!["deploy".into()];
        if let Some(image) = image {
            args.push("-i".into());
            args.push(image.to_string());
        }
        if let Some(app) = app {
            args.push("-a".into());
            args.push(app.to_string());
        }
        self.run_bin("flyctl", &args)
    }

    pub fn fly_status(&self, app: &str) -> Result<PlatformOutput> {
        self.require_bin("flyctl", "fly_status", "Install flyctl (`winget install fly-io.flyctl`).")?;
        let args = vec!["status".into(), "-a".into(), app.to_string()];
        self.read("fly_status", "flyctl status", "flyctl", &args)
    }

    // ---------------------------------------------------------------
    // Railway (`railway`)
    // ---------------------------------------------------------------

    pub fn railway_available(&self) -> bool {
        self.available("railway")
    }

    pub fn railway_whoami(&self) -> Result<PlatformOutput> {
        self.require_bin("railway", "railway_whoami", "Install Railway CLI (`npm i -g @railway/cli`).")?;
        let args = vec!["whoami".into()];
        self.read("railway_whoami", "railway whoami", "railway", &args)
    }

    pub fn railway_status(&self) -> Result<PlatformOutput> {
        self.require_bin("railway", "railway_status", "Install Railway CLI (`npm i -g @railway/cli`).")?;
        let args = vec!["status".into()];
        self.read("railway_status", "railway status", "railway", &args)
    }

    pub fn railway_up<F>(&self, detach: bool, approver: &mut F) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("railway", "railway_up", "Install Railway CLI (`npm i -g @railway/cli`).")?;
        self.enforce(
            "railway_up",
            "deploy to Railway".into(),
            "railway up".into(),
            approver,
        )?;
        let mut args = vec!["up".into()];
        if detach {
            args.push("-d".into());
        }
        self.run_bin("railway", &args)
    }

    // ---------------------------------------------------------------
    // Render (render CLI)
    // ---------------------------------------------------------------

    pub fn render_available(&self) -> bool {
        self.available("render")
    }

    pub fn render_whoami(&self) -> Result<PlatformOutput> {
        self.require_bin("render", "render_whoami", "Install Render CLI (`npm i -g @renderinc/cli`).")?;
        let args = vec!["whoami".into()];
        self.read("render_whoami", "render whoami", "render", &args)
    }

    pub fn render_services(&self) -> Result<PlatformOutput> {
        self.require_bin("render", "render_services", "Install Render CLI (`npm i -g @renderinc/cli`).")?;
        let args = vec!["services".into()];
        self.read("render_services", "render services", "render", &args)
    }

    pub fn render_deploy<F>(&self, service_id: &str, approver: &mut F) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("render", "render_deploy", "Install Render CLI (`npm i -g @renderinc/cli`).")?;
        self.enforce(
            "render_deploy",
            "trigger a deploy on Render".into(),
            "render deploys create".into(),
            approver,
        )?;
        let args = vec!["deploys".into(), "create".into(), service_id.to_string()];
        self.run_bin("render", &args)
    }

    // ---------------------------------------------------------------
    // Netlify (`netlify`)
    // ---------------------------------------------------------------

    pub fn netlify_available(&self) -> bool {
        self.available("netlify")
    }

    pub fn netlify_whoami(&self) -> Result<PlatformOutput> {
        self.require_bin("netlify", "netlify_whoami", "Install Netlify CLI (`npm i -g netlify-cli`).")?;
        let args = vec!["status".into()];
        self.read("netlify_whoami", "netlify status", "netlify", &args)
    }

    pub fn netlify_sites(&self) -> Result<PlatformOutput> {
        self.require_bin("netlify", "netlify_sites", "Install Netlify CLI (`npm i -g netlify-cli`).")?;
        let args = vec!["sites".into(), "list".into()];
        self.read("netlify_sites", "netlify sites list", "netlify", &args)
    }

    pub fn netlify_deploy<F>(&self, dir: &str, prod: bool, site: Option<&str>, approver: &mut F) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("netlify", "netlify_deploy", "Install Netlify CLI (`npm i -g netlify-cli`).")?;
        self.enforce(
            "netlify_deploy",
            "deploy to Netlify".into(),
            "netlify deploy".into(),
            approver,
        )?;
        let mut args = vec!["deploy".into(), "--dir".into(), dir.to_string()];
        if let Some(site) = site {
            args.push("--site".into());
            args.push(site.to_string());
        }
        if prod {
            args.push("--prod".into());
        }
        self.run_bin("netlify", &args)
    }

    // ---------------------------------------------------------------
    // Firebase (`firebase`)
    // ---------------------------------------------------------------

    pub fn firebase_available(&self) -> bool {
        self.available("firebase")
    }

    pub fn firebase_projects(&self) -> Result<PlatformOutput> {
        self.require_bin("firebase", "firebase_projects", "Install Firebase CLI (`npm i -g firebase-tools`).")?;
        let args = vec!["projects".into(), "list".into()];
        self.read("firebase_projects", "firebase projects list", "firebase", &args)
    }

    pub fn firebase_deploy<F>(&self, only: Option<&str>, approver: &mut F) -> Result<PlatformOutput>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        self.require_bin("firebase", "firebase_deploy", "Install Firebase CLI (`npm i -g firebase-tools`).")?;
        self.enforce(
            "firebase_deploy",
            "deploy to Firebase Hosting / Functions".into(),
            "firebase deploy".into(),
            approver,
        )?;
        let mut args = vec!["deploy".into()];
        if let Some(only) = only {
            args.push("--only".into());
            args.push(only.to_string());
        }
        self.run_bin("firebase", &args)
    }

    pub fn firebase_functions(&self) -> Result<PlatformOutput> {
        self.require_bin("firebase", "firebase_functions", "Install Firebase CLI (`npm i -g firebase-tools`).")?;
        let args = vec!["functions".into(), "list".into()];
        self.read("firebase_functions", "firebase functions list", "firebase", &args)
    }
}
