import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as opensearchserverless from 'aws-cdk-lib/aws-opensearchserverless';
import * as cloudFront from 'aws-cdk-lib/aws-cloudfront';
import * as origins from 'aws-cdk-lib/aws-cloudfront-origins';
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as path from "path";

const projectName = `strands-agent`; 
const region = process.env.CDK_DEFAULT_REGION;    
const accountId = process.env.CDK_DEFAULT_ACCOUNT;
const bucketName = `storage-for-${projectName}-${accountId}-${region}`; 
const vectorIndexName = projectName

export class CdkStrandsAgentStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // s3 
    const s3Bucket = new s3.Bucket(this, `storage-${projectName}`,{
      bucketName: bucketName,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      publicReadAccess: false,
      versioned: false,
      cors: [
        {
          allowedHeaders: ['*'],
          allowedMethods: [
            s3.HttpMethods.GET,
            s3.HttpMethods.POST,
            s3.HttpMethods.PUT,
          ],
          allowedOrigins: ['*'],
        },
      ],
    });
    new cdk.CfnOutput(this, 'bucketName', {
      value: s3Bucket.bucketName,
      description: 'The nmae of bucket',
    });

    // cloudfront for sharing s3
    const distribution_sharing = new cloudFront.Distribution(this, `sharing-for-${projectName}`, {
      defaultBehavior: {
        origin: origins.S3BucketOrigin.withOriginAccessControl(s3Bucket),
        allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
        cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
        viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
      },
      priceClass: cloudFront.PriceClass.PRICE_CLASS_200,  
    });
    new cdk.CfnOutput(this, `distribution-sharing-DomainName-for-${projectName}`, {
      value: 'https://'+distribution_sharing.domainName,
      description: 'The domain name of the Distribution Sharing',
    });   

    // Knowledge Base Role
    const roleKnowledgeBase = new iam.Role(this,  `role-knowledge-base-for-${projectName}`, {
      roleName: `role-knowledge-base-for-${projectName}-${region}`,
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal("bedrock.amazonaws.com")
      )
    });
    
    const bedrockInvokePolicy = new iam.PolicyStatement({ 
      effect: iam.Effect.ALLOW,
      resources: [`*`],
      actions: ["bedrock:*"],
    });        
    roleKnowledgeBase.attachInlinePolicy( 
      new iam.Policy(this, `bedrock-invoke-policy-for-${projectName}`, {
        statements: [bedrockInvokePolicy],
      }),
    );  
    
    const S3Policy = new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      resources: ['*'],
      actions: ["s3:*"],
    });
    roleKnowledgeBase.attachInlinePolicy( 
      new iam.Policy(this, `s3-policy-knowledge-base-for-${projectName}`, {
        statements: [S3Policy],
      }),
    );     
    const bedrockKnowledgeBaseS3Policy = new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      resources: ['*'],
      actions: ["s3:*"],
    });
    roleKnowledgeBase.attachInlinePolicy( 
      new iam.Policy(this, `knowledge-base-s3-policy-for-${projectName}`, {
        statements: [bedrockKnowledgeBaseS3Policy],
      }),
    );      
    const knowledgeBaseOpenSearchPolicy = new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      resources: ['*'],
      actions: ["aoss:APIAccessAll"],
    });
    roleKnowledgeBase.attachInlinePolicy( 
      new iam.Policy(this, `bedrock-agent-opensearch-policy-for-${projectName}`, {
        statements: [knowledgeBaseOpenSearchPolicy],
      }),
    );  

    // lambda Knowledge Base
    const roleLambdaKnowledgeBase = new iam.Role(this, `role-lambda-knowledge-base-for-${projectName}`, {
      roleName: `role-lambda-knowledge-base-for-${projectName}-${region}`,
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal("lambda.amazonaws.com"),
        new iam.ServicePrincipal("bedrock.amazonaws.com"),
      ),
    });
    const CreateLogPolicy = new iam.PolicyStatement({  
      resources: [`arn:aws:logs:${region}:${accountId}:*`],
      actions: ['logs:CreateLogGroup'],
    });        
    roleLambdaKnowledgeBase.attachInlinePolicy( 
      new iam.Policy(this, `log-policy-lambda-knowledge-base-for-${projectName}`, {
        statements: [CreateLogPolicy],
      }),
    );
    const CreateLogStreamPolicy = new iam.PolicyStatement({  
      resources: [`arn:aws:logs:${region}:${accountId}:log-group:/aws/lambda/*`],
      actions: ["logs:CreateLogStream","logs:PutLogEvents"],
    });        
    roleLambdaKnowledgeBase.attachInlinePolicy( 
      new iam.Policy(this, `stream-log-policy-lambda-knowledge-base-for-${projectName}`, {
        statements: [CreateLogStreamPolicy],
      }),
    );

    const knowledgeBaseBedrockPolicy = new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      resources: ['*'],
      actions: ["bedrock:*"],
    });
    roleKnowledgeBase.attachInlinePolicy( 
      new iam.Policy(this, `bedrock-policy-knowledge-base-for-${projectName}`, {
        statements: [knowledgeBaseBedrockPolicy],
      }),
    );  
    roleLambdaKnowledgeBase.attachInlinePolicy( 
      new iam.Policy(this, `bedrock-policy-lambda-knowledge-base-for-${projectName}`, {
        statements: [knowledgeBaseBedrockPolicy],
      }),
    );  
    roleLambdaKnowledgeBase.attachInlinePolicy( 
      new iam.Policy(this, `s3-policy-lambda-knowledge-base-for-${projectName}`, {
        statements: [S3Policy],
      }),
    );

    // OpenSearch Serverless
    const collectionName = vectorIndexName
    const OpenSearchCollection = new opensearchserverless.CfnCollection(this, `opensearch-correction-for-${projectName}`, {
      name: collectionName,    
      description: `opensearch correction for ${projectName}`,
      standbyReplicas: 'DISABLED',
      type: 'VECTORSEARCH',
    });
    const collectionArn = OpenSearchCollection.attrArn

    new cdk.CfnOutput(this, `OpensearchCollectionEndpoint-${projectName}`, {
      value: OpenSearchCollection.attrCollectionEndpoint,
      description: 'The endpoint of opensearch correction',
    });

    const encPolicyName = `encription-${projectName}`
    const encPolicy = new opensearchserverless.CfnSecurityPolicy(this, `opensearch-encription-policy-for-${projectName}`, {
      name: encPolicyName,
      type: "encryption",
      description: `opensearch encryption policy for ${projectName}`,
      policy:
        `{"Rules":[{"ResourceType":"collection","Resource":["collection/${collectionName}"]}],"AWSOwnedKey":true}`,
    });
    OpenSearchCollection.addDependency(encPolicy);

    const netPolicyName = `network-${projectName}-${region}`
    const netPolicy = new opensearchserverless.CfnSecurityPolicy(this, `opensearch-network-policy-for-${projectName}`, {
      name: netPolicyName,
      type: 'network',    
      description: `opensearch network policy for ${projectName}`,
      policy: JSON.stringify([
        {
          Rules: [
            {
              ResourceType: "dashboard",
              Resource: [`collection/${collectionName}`],
            },
            {
              ResourceType: "collection",
              Resource: [`collection/${collectionName}`],              
            }
          ],
          AllowFromPublic: true,          
        },
      ]), 
    });
    OpenSearchCollection.addDependency(netPolicy);

    const account = new iam.AccountPrincipal(this.account)
    const dataAccessPolicyName = `data-${projectName}`
    const dataAccessPolicy = new opensearchserverless.CfnAccessPolicy(this, `opensearch-data-collection-policy-for-${projectName}`, {
      name: dataAccessPolicyName,
      type: "data",
      policy: JSON.stringify([
        {
          Rules: [
            {
              Resource: [`collection/${collectionName}`],
              Permission: [
                "aoss:CreateCollectionItems",
                "aoss:DeleteCollectionItems",
                "aoss:UpdateCollectionItems",
                "aoss:DescribeCollectionItems",
              ],
              ResourceType: "collection",
            },
            {
              Resource: [`index/${collectionName}/*`],
              Permission: [
                "aoss:CreateIndex",
                "aoss:DeleteIndex",
                "aoss:UpdateIndex",
                "aoss:DescribeIndex",
                "aoss:ReadDocument",
                "aoss:WriteDocument",
              ], 
              ResourceType: "index",
            }
          ],
          Principal: [
            account.arn
          ], 
        },
      ]),
    });
    OpenSearchCollection.addDependency(dataAccessPolicy);

    // Weather
    new secretsmanager.Secret(this, `weather-api-secret-for-${projectName}`, {
      description: 'secret for weather api key', // openweathermap
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      secretName: `openweathermap-${projectName}`,
      secretObjectValue: {
        project_name: cdk.SecretValue.unsafePlainText(projectName),
        weather_api_key: cdk.SecretValue.unsafePlainText(''),
      },
    });

    // Tavily
    new secretsmanager.Secret(this, `tavily-secret-for-${projectName}`, {
      description: 'secret for tavily api key', // tavily
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      secretName: `tavilyapikey-${projectName}`,
      secretObjectValue: {
        project_name: cdk.SecretValue.unsafePlainText(projectName),
        tavily_api_key: cdk.SecretValue.unsafePlainText(''),
      },
    });

    // perplexity
    new secretsmanager.Secret(this, `perflexity-secret-for-${projectName}`, {
      description: 'secret for perflexity api key', // tavily
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      secretName: `perplexityapikey-${projectName}`,
      secretObjectValue: {
        project_name: cdk.SecretValue.unsafePlainText(projectName),
        perplexity_api_key: cdk.SecretValue.unsafePlainText(''),
      },
    });

    // firecrawl
    new secretsmanager.Secret(this, `firecrawl-secret-for-${projectName}`, {
      description: 'secret for firecrawl api key', // firecrawl
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      secretName: `firecrawlapikey-${projectName}`,
      secretObjectValue: {
        project_name: cdk.SecretValue.unsafePlainText(projectName),
        firecrawl_api_key: cdk.SecretValue.unsafePlainText(''),
      },
    });

    const novaActSecret = new secretsmanager.Secret(this, `nova-act-secret-for-${projectName}`, {
      description: 'secret for nova act api key', // nova act
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      secretName: `nova-act-${projectName}`,
      secretObjectValue: {
        project_name: cdk.SecretValue.unsafePlainText(projectName),
        nova_act_api_key: cdk.SecretValue.unsafePlainText(''),
      },
    });
    
    // lambda-rag
    const roleLambdaRag = new iam.Role(this, `role-lambda-rag-for-${projectName}`, {
      roleName: `role-lambda-rag-for-${projectName}-${region}`,
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal("lambda.amazonaws.com"),
        new iam.ServicePrincipal("bedrock.amazonaws.com"),
      ),
      // managedPolicies: [cdk.aws_iam.ManagedPolicy.fromAwsManagedPolicyName('CloudWatchLogsFullAccess')] 
    });
    roleLambdaRag.attachInlinePolicy( 
      new iam.Policy(this, `create-log-policy-lambda-rag-for-${projectName}`, {
        statements: [CreateLogPolicy],
      }),
    );
    roleLambdaRag.attachInlinePolicy( 
      new iam.Policy(this, `create-stream-log-policy-lambda-rag-for-${projectName}`, {
        statements: [CreateLogStreamPolicy],
      }),
    );      

    // bedrock
    roleLambdaRag.attachInlinePolicy( 
      new iam.Policy(this, `tool-bedrock-invoke-policy-for-${projectName}`, {
        statements: [bedrockInvokePolicy],
      }),
    );  
    roleLambdaRag.attachInlinePolicy( 
      new iam.Policy(this, `tool-bedrock-agent-opensearch-policy-for-${projectName}`, {
        statements: [knowledgeBaseOpenSearchPolicy],
      }),
    );  
    roleLambdaRag.attachInlinePolicy( 
      new iam.Policy(this, `tool-bedrock-agent-bedrock-policy-for-${projectName}`, {
        statements: [knowledgeBaseBedrockPolicy],
      }),
    );  
    
    const lambdaKnowledgeBase = new lambda.DockerImageFunction(this, `knowledge-base-for-${projectName}`, {
      description: 'RAG based on Knoeledge Base',
      functionName: `knowledge-base-for-${projectName}`,
      code: lambda.DockerImageCode.fromImageAsset(path.join(__dirname, '../../lambda-knowledge-base')),
      timeout: cdk.Duration.seconds(120),
      memorySize: 4096,
      role: roleLambdaRag,
      environment: {
        bedrock_region: String(region),  
        projectName: projectName,
        "sharing_url": 'https://'+distribution_sharing.domainName,
      }
    });
    lambdaKnowledgeBase.grantInvoke(new cdk.aws_iam.ServicePrincipal("bedrock.amazonaws.com"));     
    
    const environment = {
      "projectName": projectName,
      "accountId": accountId,
      "region": region,
      "knowledge_base_role": roleKnowledgeBase.roleArn,
      "collectionArn": collectionArn,
      "opensearch_url": OpenSearchCollection.attrCollectionEndpoint,
      "s3_bucket": s3Bucket.bucketName,      
      "s3_arn": s3Bucket.bucketArn,
      "sharing_url": 'https://'+distribution_sharing.domainName,
    }    
    new cdk.CfnOutput(this, `environment-for-${projectName}`, {
      value: JSON.stringify(environment),
      description: `environment-${projectName}`,
      exportName: `environment-${projectName}`
    });    
  }
}
