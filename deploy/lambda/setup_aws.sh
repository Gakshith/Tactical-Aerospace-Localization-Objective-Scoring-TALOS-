#!/usr/bin/env bash
# One-time AWS setup for the TALOS Lambda, run with an admin AWS CLI login:
#   REGION=us-east-1 bash deploy/lambda/setup_aws.sh
# Creates (idempotently):
#   * ECR repo talos-lambda, keeping only the last 3 images (storage stays tiny)
#   * GitHub OIDC identity provider (no long-lived AWS keys in GitHub)
#   * role talos-github-deploy: assumable only by this repo's main branch; may push to
#     that one ECR repo and create/update the one Lambda function "talos"
#   * role talos-lambda-exec: what the function runs as (CloudWatch logs only)
# Then stores AWS_ROLE_ARN / AWS_REGION as GitHub repo variables, so the
# "Deploy Lambda" workflow can run. The workflow creates the function itself.
set -euo pipefail
REGION=${REGION:-us-east-1}
GH_REPO=${GH_REPO:-Gakshith/Tactical-Aerospace-Localization-Objective-Scoring-TALOS-}
ECR_REPO=talos-lambda
FUNC=talos
export AWS_REGION=$REGION AWS_DEFAULT_REGION=$REGION
ACCT=$(aws sts get-caller-identity --query Account --output text)
echo "account ...${ACCT: -4}, region $REGION"

# ECR
aws ecr describe-repositories --repository-names $ECR_REPO >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name $ECR_REPO --image-scanning-configuration scanOnPush=true >/dev/null
aws ecr put-lifecycle-policy --repository-name $ECR_REPO --lifecycle-policy-text '{"rules":[{"rulePriority":1,
  "description":"keep last 3","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":3},
  "action":{"type":"expire"}}]}' >/dev/null

# GitHub OIDC provider
OIDC=arn:aws:iam::$ACCT:oidc-provider/token.actions.githubusercontent.com
aws iam get-open-id-connect-provider --open-id-connect-provider-arn $OIDC >/dev/null 2>&1 \
  || aws iam create-open-id-connect-provider --url https://token.actions.githubusercontent.com \
       --client-id-list sts.amazonaws.com >/dev/null

# Lambda execution role
aws iam get-role --role-name talos-lambda-exec >/dev/null 2>&1 || {
  aws iam create-role --role-name talos-lambda-exec --assume-role-policy-document '{"Version":"2012-10-17",
    "Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name talos-lambda-exec \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
}

# GitHub deploy role: main branch of this repo only
TRUST=$(cat <<JSON
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Federated":"$OIDC"},
 "Action":"sts:AssumeRoleWithWebIdentity","Condition":{
  "StringEquals":{"token.actions.githubusercontent.com:aud":"sts.amazonaws.com",
                  "token.actions.githubusercontent.com:sub":"repo:$GH_REPO:ref:refs/heads/main"}}}]}
JSON
)
aws iam get-role --role-name talos-github-deploy >/dev/null 2>&1 \
  && aws iam update-assume-role-policy --role-name talos-github-deploy --policy-document "$TRUST" \
  || aws iam create-role --role-name talos-github-deploy --assume-role-policy-document "$TRUST" >/dev/null
FN_ARN=arn:aws:lambda:$REGION:$ACCT:function:$FUNC
aws iam put-role-policy --role-name talos-github-deploy --policy-name talos-deploy --policy-document "$(cat <<JSON
{"Version":"2012-10-17","Statement":[
 {"Effect":"Allow","Action":"ecr:GetAuthorizationToken","Resource":"*"},
 {"Effect":"Allow","Action":["ecr:BatchCheckLayerAvailability","ecr:BatchGetImage","ecr:CompleteLayerUpload",
   "ecr:GetDownloadUrlForLayer","ecr:InitiateLayerUpload","ecr:PutImage","ecr:UploadLayerPart"],
  "Resource":"arn:aws:ecr:$REGION:$ACCT:repository/$ECR_REPO"},
 {"Effect":"Allow","Action":["lambda:GetFunction","lambda:GetFunctionConfiguration","lambda:CreateFunction",
   "lambda:UpdateFunctionCode","lambda:UpdateFunctionConfiguration","lambda:GetFunctionUrlConfig",
   "lambda:CreateFunctionUrlConfig","lambda:GetPolicy","lambda:AddPermission"],"Resource":"$FN_ARN"},
 {"Effect":"Allow","Action":"iam:PassRole","Resource":"arn:aws:iam::$ACCT:role/talos-lambda-exec"}]}
JSON
)"

gh variable set AWS_ROLE_ARN --repo "$GH_REPO" --body "arn:aws:iam::$ACCT:role/talos-github-deploy"
gh variable set AWS_REGION --repo "$GH_REPO" --body "$REGION"
echo "done. Run the 'Deploy Lambda' workflow (or push to main) to build and deploy."
