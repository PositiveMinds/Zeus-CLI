//! Platform-CLI tool dispatch (gh, supabase, vercel, docker, kubectl, terraform, circleci, aws, az, gcloud...).

use super::*;

impl ToolManager {
    /// Dispatch a platform-CLI tool (gh/supabase/vercel/docker/kubectl/
    /// terraform/circleci) to the matching `PlatformEngine` method. Arguments
    /// are read from the JSON tool args; read-only ops ignore the approver.
    pub(super) fn do_platform<F>(
        &self,
        name: &str,
        args: &Value,
        approver: &mut F,
    ) -> Result<ToolResult>
    where
        F: FnMut(&PermissionRequest) -> ApprovalDecision,
    {
        let count = |k: &str| Self::usize_arg(args, k).unwrap_or(20);
        let result = match name {
            "gh_issue_list" => {
                let label = Self::opt_str_arg(args, "label");
                let state = Self::opt_str_arg(args, "state").unwrap_or("open");
                platform_result(self.platform.gh_issue_list(state, count("limit"), label))
            }
            "gh_issue_view" => {
                let n = Self::str_arg(args, "number")?;
                platform_result(self.platform.gh_issue_view(n))
            }
            "gh_issue_create" => {
                let title = Self::str_arg(args, "title")?;
                let body = Self::opt_str_arg(args, "body");
                let label = Self::opt_str_arg(args, "label");
                platform_result(
                    self.platform
                        .gh_issue_create(title, body, label, &mut *approver),
                )
            }
            "gh_issue_close" => {
                let n = Self::str_arg(args, "number")?;
                platform_result(self.platform.gh_issue_close(n, &mut *approver))
            }
            "gh_pr_list" => {
                let state = Self::opt_str_arg(args, "state").unwrap_or("open");
                platform_result(self.platform.gh_pr_list(state, count("limit")))
            }
            "gh_pr_view" => {
                let n = Self::str_arg(args, "number")?;
                platform_result(self.platform.gh_pr_view(n))
            }
            "gh_pr_create" => {
                let title = Self::str_arg(args, "title")?;
                let body = Self::opt_str_arg(args, "body");
                let base = Self::opt_str_arg(args, "base");
                platform_result(
                    self.platform
                        .gh_pr_create(title, body, base, &mut *approver),
                )
            }
            "gh_pr_merge" => {
                let n = Self::str_arg(args, "number")?;
                let method = Self::opt_str_arg(args, "method");
                let del = Self::opt_bool_arg(args, "delete_branch").unwrap_or(false);
                platform_result(self.platform.gh_pr_merge(n, method, del, &mut *approver))
            }
            "gh_release_list" => platform_result(self.platform.gh_release_list(count("limit"))),
            "gh_release_create" => {
                let tag = Self::str_arg(args, "tag")?;
                let title = Self::opt_str_arg(args, "title");
                let notes = Self::opt_str_arg(args, "notes");
                platform_result(
                    self.platform
                        .gh_release_create(tag, title, notes, &mut *approver),
                )
            }
            "gh_workflow_list" => platform_result(self.platform.gh_workflow_list()),
            "gh_workflow_run" => {
                let wf = Self::str_arg(args, "workflow")?;
                let r = Self::opt_str_arg(args, "ref");
                platform_result(self.platform.gh_workflow_run(wf, r, &mut *approver))
            }
            "gh_run_list" => {
                let wf = Self::opt_str_arg(args, "workflow");
                platform_result(self.platform.gh_run_list(wf, count("limit")))
            }
            "supabase_login" => platform_result(self.platform.supabase_login(&mut *approver)),
            "supabase_link" => {
                let pr = Self::opt_str_arg(args, "project_ref");
                platform_result(self.platform.supabase_link(pr, &mut *approver))
            }
            "supabase_projects_list" => platform_result(self.platform.supabase_projects_list()),
            "supabase_status" => platform_result(self.platform.supabase_status()),
            "supabase_db_push" => platform_result(self.platform.supabase_db_push(&mut *approver)),
            "supabase_db_diff" => {
                let schema = Self::opt_str_arg(args, "schema");
                let linked = Self::opt_bool_arg(args, "linked").unwrap_or(false);
                platform_result(self.platform.supabase_db_diff(schema, linked))
            }
            "supabase_functions_list" => platform_result(self.platform.supabase_functions_list()),
            "supabase_functions_deploy" => {
                let f = Self::str_arg(args, "function")?;
                let pr = Self::opt_str_arg(args, "project_ref");
                let nvj = Self::opt_bool_arg(args, "no_verify_jwt").unwrap_or(false);
                platform_result(
                    self.platform
                        .supabase_functions_deploy(f, pr, nvj, &mut *approver),
                )
            }
            "vercel_whoami" => platform_result(self.platform.vercel_whoami()),
            "vercel_projects_list" => platform_result(self.platform.vercel_projects_list()),
            "vercel_env_list" => {
                let env = Self::opt_str_arg(args, "env");
                let project = Self::opt_str_arg(args, "project");
                platform_result(self.platform.vercel_env_list(env, project))
            }
            "vercel_deploy" => {
                let prod = Self::opt_bool_arg(args, "prod").unwrap_or(false);
                let target = Self::opt_str_arg(args, "target");
                let project = Self::opt_str_arg(args, "project");
                platform_result(
                    self.platform
                        .vercel_deploy(prod, target, project, &mut *approver),
                )
            }
            "vercel_logs" => {
                let dep = Self::opt_str_arg(args, "deployment");
                let project = Self::opt_str_arg(args, "project");
                let follow = Self::opt_bool_arg(args, "follow").unwrap_or(false);
                platform_result(self.platform.vercel_logs(dep, project, follow))
            }
            "docker_ps" => {
                let all = Self::opt_bool_arg(args, "all").unwrap_or(false);
                platform_result(self.platform.docker_ps(all))
            }
            "docker_images" => platform_result(self.platform.docker_images()),
            "docker_compose_up" => {
                let services = Self::str_array_arg(args, "services");
                let detached = Self::opt_bool_arg(args, "detached").unwrap_or(false);
                let build = Self::opt_bool_arg(args, "build").unwrap_or(false);
                platform_result(self.platform.docker_compose_up(
                    services,
                    detached,
                    build,
                    &mut *approver,
                ))
            }
            "docker_compose_down" => {
                let volumes = Self::opt_bool_arg(args, "volumes").unwrap_or(false);
                platform_result(self.platform.docker_compose_down(volumes, &mut *approver))
            }
            "docker_compose_logs" => {
                let service = Self::opt_str_arg(args, "service");
                let follow = Self::opt_bool_arg(args, "follow").unwrap_or(false);
                platform_result(self.platform.docker_compose_logs(service, follow))
            }
            "k8s_get" => {
                let resource = Self::str_arg(args, "resource")?;
                let n = Self::opt_str_arg(args, "name");
                let ns = Self::opt_str_arg(args, "namespace");
                let an = Self::opt_bool_arg(args, "all_namespaces").unwrap_or(false);
                platform_result(self.platform.k8s_get(resource, n, ns, an))
            }
            "k8s_logs" => {
                let pod = Self::str_arg(args, "pod")?;
                let c = Self::opt_str_arg(args, "container");
                let ns = Self::opt_str_arg(args, "namespace");
                let follow = Self::opt_bool_arg(args, "follow").unwrap_or(false);
                platform_result(self.platform.k8s_logs(pod, c, ns, follow))
            }
            "k8s_apply" => {
                let path = Self::str_arg(args, "path")?;
                let ns = Self::opt_str_arg(args, "namespace");
                platform_result(self.platform.k8s_apply(path, ns, &mut *approver))
            }
            "k8s_rollout_status" => {
                let resource = Self::str_arg(args, "resource")?;
                let ns = Self::opt_str_arg(args, "namespace");
                platform_result(self.platform.k8s_rollout_status(resource, ns))
            }
            "tf_init" => platform_result(self.platform.tf_init(&mut *approver)),
            "tf_validate" => platform_result(self.platform.tf_validate()),
            "tf_plan" => {
                let out = Self::opt_str_arg(args, "out");
                platform_result(self.platform.tf_plan(out))
            }
            "tf_apply" => {
                let plan = Self::opt_str_arg(args, "plan_file");
                let aa = Self::opt_bool_arg(args, "auto_approve").unwrap_or(false);
                platform_result(self.platform.tf_apply(plan, aa, &mut *approver))
            }
            "circleci_validate" => {
                let cfg = Self::opt_str_arg(args, "config");
                platform_result(self.platform.circleci_validate(cfg))
            }
            "circleci_builds" => {
                let project = Self::str_arg(args, "project")?;
                let branch = Self::opt_str_arg(args, "branch");
                platform_result(
                    self.platform
                        .circleci_builds(project, branch, count("limit")),
                )
            }
            "aws_whoami" => platform_result(self.platform.aws_whoami()),
            "aws_s3_ls" => {
                let path = Self::opt_str_arg(args, "path");
                platform_result(self.platform.aws_s3_ls(path))
            }
            "aws_s3_sync" => {
                let source = Self::str_arg(args, "source")?;
                let dest = Self::str_arg(args, "dest")?;
                platform_result(self.platform.aws_s3_sync(source, dest, &mut *approver))
            }
            "aws_ecr_login" => platform_result(self.platform.aws_ecr_login(&mut *approver)),
            "aws_lambda_list" => platform_result(self.platform.aws_lambda_list()),
            "aws_lambda_invoke" => {
                let function = Self::str_arg(args, "function")?;
                let payload = Self::opt_str_arg(args, "payload");
                platform_result(
                    self.platform
                        .aws_lambda_invoke(function, payload, &mut *approver),
                )
            }
            "aws_ecs_list_clusters" => platform_result(self.platform.aws_ecs_list_clusters()),
            "aws_ecs_force_deploy" => {
                let cluster = Self::str_arg(args, "cluster")?;
                let service = Self::str_arg(args, "service")?;
                platform_result(self.platform.aws_ecs_force_deploy(
                    cluster,
                    service,
                    &mut *approver,
                ))
            }
            "sam_build" => platform_result(self.platform.sam_build(&mut *approver)),
            "sam_deploy" => {
                let guided = Self::opt_bool_arg(args, "guided").unwrap_or(false);
                let stack_name = Self::opt_str_arg(args, "stack_name");
                platform_result(self.platform.sam_deploy(guided, stack_name, &mut *approver))
            }
            "cloudformation_describe" => {
                let stack = Self::str_arg(args, "stack")?;
                platform_result(self.platform.cloudformation_describe(stack))
            }
            "cloudformation_deploy" => {
                let template = Self::str_arg(args, "template")?;
                let stack = Self::str_arg(args, "stack")?;
                platform_result(self.platform.cloudformation_deploy(
                    template,
                    stack,
                    &mut *approver,
                ))
            }
            "az_whoami" => platform_result(self.platform.az_whoami()),
            "az_webapp_list" => platform_result(self.platform.az_webapp_list()),
            "az_webapp_deploy" => {
                let name = Self::str_arg(args, "name")?;
                let rg = Self::str_arg(args, "resource_group")?;
                let source = Self::str_arg(args, "source")?;
                platform_result(
                    self.platform
                        .az_webapp_deploy(name, rg, source, &mut *approver),
                )
            }
            "az_functionapp_deploy" => {
                let name = Self::str_arg(args, "name")?;
                let rg = Self::str_arg(args, "resource_group")?;
                let source = Self::str_arg(args, "source")?;
                platform_result(self.platform.az_functionapp_deploy(
                    name,
                    rg,
                    source,
                    &mut *approver,
                ))
            }
            "gcloud_whoami" => platform_result(self.platform.gcloud_whoami()),
            "gcloud_app_deploy" => platform_result(self.platform.gcloud_app_deploy(&mut *approver)),
            "gcloud_run_deploy" => {
                let service = Self::str_arg(args, "service")?;
                let image = Self::str_arg(args, "image")?;
                let region = Self::opt_str_arg(args, "region");
                platform_result(self.platform.gcloud_run_deploy(
                    service,
                    image,
                    region,
                    &mut *approver,
                ))
            }
            "gcloud_run_services" => platform_result(self.platform.gcloud_run_services()),
            "helm_list" => {
                let ns = Self::opt_str_arg(args, "namespace");
                platform_result(self.platform.helm_list(ns))
            }
            "helm_status" => {
                let release = Self::str_arg(args, "release")?;
                let ns = Self::opt_str_arg(args, "namespace");
                platform_result(self.platform.helm_status(release, ns))
            }
            "helm_install" => {
                let release = Self::str_arg(args, "release")?;
                let chart = Self::str_arg(args, "chart")?;
                let ns = Self::opt_str_arg(args, "namespace");
                platform_result(
                    self.platform
                        .helm_install(release, chart, ns, &mut *approver),
                )
            }
            "helm_upgrade" => {
                let release = Self::str_arg(args, "release")?;
                let chart = Self::str_arg(args, "chart")?;
                let ns = Self::opt_str_arg(args, "namespace");
                platform_result(
                    self.platform
                        .helm_upgrade(release, chart, ns, &mut *approver),
                )
            }
            "helm_uninstall" => {
                let release = Self::str_arg(args, "release")?;
                let ns = Self::opt_str_arg(args, "namespace");
                platform_result(self.platform.helm_uninstall(release, ns, &mut *approver))
            }
            "fly_whoami" => platform_result(self.platform.fly_whoami()),
            "fly_apps_list" => platform_result(self.platform.fly_apps_list()),
            "fly_deploy" => {
                let image = Self::opt_str_arg(args, "image");
                let app = Self::opt_str_arg(args, "app");
                platform_result(self.platform.fly_deploy(image, app, &mut *approver))
            }
            "fly_status" => {
                let app = Self::str_arg(args, "app")?;
                platform_result(self.platform.fly_status(app))
            }
            "railway_whoami" => platform_result(self.platform.railway_whoami()),
            "railway_status" => platform_result(self.platform.railway_status()),
            "railway_up" => {
                let detach = Self::opt_bool_arg(args, "detach").unwrap_or(false);
                platform_result(self.platform.railway_up(detach, &mut *approver))
            }
            "render_whoami" => platform_result(self.platform.render_whoami()),
            "render_services" => platform_result(self.platform.render_services()),
            "render_deploy" => {
                let service_id = Self::str_arg(args, "service_id")?;
                platform_result(self.platform.render_deploy(service_id, &mut *approver))
            }
            "netlify_whoami" => platform_result(self.platform.netlify_whoami()),
            "netlify_sites" => platform_result(self.platform.netlify_sites()),
            "netlify_deploy" => {
                let dir = Self::str_arg(args, "dir")?;
                let prod = Self::opt_bool_arg(args, "prod").unwrap_or(false);
                let site = Self::opt_str_arg(args, "site");
                platform_result(
                    self.platform
                        .netlify_deploy(dir, prod, site, &mut *approver),
                )
            }
            "firebase_projects" => platform_result(self.platform.firebase_projects()),
            "firebase_deploy" => {
                let only = Self::opt_str_arg(args, "only");
                platform_result(self.platform.firebase_deploy(only, &mut *approver))
            }
            "firebase_functions" => platform_result(self.platform.firebase_functions()),
            _ => return Err(AgentError::UnknownTool(name.to_string())),
        };
        result
    }
}

/// Same convention as `git_result` for the platform-CLI engines.
fn platform_result(result: zeus_fs::Result<PlatformOutput>) -> Result<ToolResult> {
    match result {
        Ok(out) => {
            let text = format!(
                "exit={:?}\n--- stdout ---\n{}--- stderr ---\n{}",
                out.exit_code, out.stdout, out.stderr
            );
            if out.success {
                Ok(ToolResult::ok(text))
            } else {
                Ok(ToolResult::err(text))
            }
        }
        Err(e) => Ok(ToolResult::err(e.to_string())),
    }
}
