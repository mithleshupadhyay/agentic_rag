# Authentication And Tenant Access

## Production Boundary

Agentic RAG uses Keycloak as its OpenID Connect identity service. Keycloak can
broker Google and GitHub while also supporting invited users who sign in with
email or username and password.

```text
Browser -> Keycloak -> password, Google, or GitHub authentication
Browser -> Agentic RAG API with a Keycloak access token
API -> PostgreSQL tenant membership, role, scope, and ACL authorization
```

Keycloak stores salted one-way password hashes in its PostgreSQL database.
Agentic RAG does not receive a password and does not store password hashes in
its `users` table. That table stores tenant membership, the Keycloak subject,
email, status, role links, workspace context, and ACL version.

Use a dedicated `keycloak` database or schema and database user on the same
PostgreSQL cluster. Do not mix Keycloak tables into the Agentic RAG application
schema.

## 1. Deploy Keycloak With PostgreSQL

Configure a production Keycloak deployment with values equivalent to:

```dotenv
KC_DB=postgres
KC_DB_URL=jdbc:postgresql://postgres:5432/keycloak
KC_DB_USERNAME=keycloak
KC_DB_PASSWORD=replace-with-a-secret
KC_HOSTNAME=https://identity.example.com
KC_PROXY_HEADERS=xforwarded
```

Create a realm named `agentic-rag`. In Realm settings -> Login:

- Keep user registration disabled. Tenant administrators invite users.
- Enable login with email.
- Keep duplicate emails disabled.
- Enable forgot password.
- Enable verify email.

## 2. Create The Browser Client

Create an OpenID Connect client named `agentic-rag-web`:

- Client authentication: off
- Standard flow: on
- Direct access grants: off
- PKCE method: S256
- Valid redirect URI: `https://app.example.com/auth/callback`
- Valid post-logout redirect URI: `https://app.example.com/`
- Web origin: `https://app.example.com`

For local development, also allow:

```text
http://localhost:5173/auth/callback
http://localhost:5173/
```

## 3. Create The API Client And Roles

Create an OpenID Connect client named `agentic-rag-api`. Add these client
roles exactly as written:

```text
documents:read
documents:write
documents:delete
query:run
ingestion:write
```

Create these realm roles:

```text
viewer
user
admin
```

Add an audience mapper to the browser client so its access tokens contain
`agentic-rag-api` in the `aud` claim.

The invitation service maps fixed permissions by tenant role:

| Tenant role | API permissions |
|---|---|
| `viewer` | `documents:read`, `query:run` |
| `user` | viewer permissions plus `documents:write`, `ingestion:write` |
| `admin` | user permissions plus `documents:delete` |

## 4. Configure Tenant Claims

In Realm settings -> User profile, add these attributes. Administrators may
edit them; users must not be allowed to change them:

```text
tenant_id
workspace_id
acl_version
```

Add protocol mappers to the browser client or a dedicated client scope:

| User attribute | Token claim | Access token type |
|---|---|---|
| `tenant_id` | `tenant_id` | String |
| `workspace_id` | `workspace_id` | String |
| `acl_version` | `acl_version` | int |

Add the built-in groups mapper if document ACLs use Keycloak groups.

The API rejects tokens without `tenant_id`. It then loads the same subject and
tenant from PostgreSQL. Database membership status, role, workspace, and ACL
version are authoritative even if a token contains broader roles.

## 5. Create The Confidential Admin Client

Create an OpenID Connect client named `agentic-rag-admin`:

- Client authentication: on
- Service accounts roles: on
- Standard flow: off
- Direct access grants: off

Copy its client secret into `KEYCLOAK_ADMIN_CLIENT_SECRET`. Give the service
account only the administration permissions needed to create and delete users,
view clients and roles, map the configured roles, and send required-action
emails. Standard `realm-management` roles normally include:

```text
manage-users
view-users
query-users
view-realm
view-clients
```

Keycloak versions with fine-grained admin permissions must also allow this
service account to map the three realm roles and the five `agentic-rag-api`
client roles. Do not retain the broad `realm-admin` role in production.

## 6. Configure Gmail SMTP

For Gmail, enable two-step verification and create a Google App Password for
Keycloak. Do not use the normal Gmail account password.

In Keycloak Realm settings -> Email, configure:

```text
From: your-address@gmail.com
Host: smtp.gmail.com
Port: 587
Encryption: STARTTLS
Authentication: enabled
Username: your-address@gmail.com
Password: the Google App Password
```

Use Test connection before sending an invitation.

## 7. Configure Google Login

In Google Cloud Console:

1. Configure the OAuth consent screen.
2. Create an OAuth 2.0 Client ID of type Web application.
3. Add this exact authorized redirect URI:

```text
https://identity.example.com/realms/agentic-rag/broker/google/endpoint
```

In Keycloak Identity providers -> Google, set alias `google` and enter the
Google client ID and client secret.

## 8. Configure GitHub Login

In GitHub Settings -> Developer settings -> OAuth Apps:

1. Create a new OAuth App.
2. Set Homepage URL to `https://app.example.com`.
3. Set Authorization callback URL to:

```text
https://identity.example.com/realms/agentic-rag/broker/github/endpoint
```

In Keycloak Identity providers -> GitHub, set alias `github` and enter the
GitHub client ID and client secret.

Keep Keycloak's secure first-broker-login account-linking flow. An invited user
must complete email verification and password setup first. When the same user
later chooses Google or GitHub, Keycloak detects the existing email and requires
account confirmation instead of silently merging identities.

## 9. Configure Agentic RAG

Set these values in `.env`:

```dotenv
AUTH_PROVIDER=keycloak
OIDC_ISSUER_URL=https://identity.example.com/realms/agentic-rag
OIDC_AUDIENCE=agentic-rag-api
OIDC_JWKS_URL=
OIDC_FRONTEND_CLIENT_ID=agentic-rag-web
OIDC_FRONTEND_SCOPE=openid profile email
OIDC_IDENTITY_PROVIDERS=google,github
OIDC_REQUIRE_TENANT_CLAIM=true
OIDC_REQUIRE_DATABASE_MEMBERSHIP=true

KEYCLOAK_ADMIN_BASE_URL=https://identity.example.com
KEYCLOAK_ADMIN_REALM=agentic-rag
KEYCLOAK_ADMIN_CLIENT_ID=agentic-rag-admin
KEYCLOAK_ADMIN_CLIENT_SECRET=replace-with-the-client-secret
KEYCLOAK_ADMIN_TIMEOUT_SECONDS=10
KEYCLOAK_INVITATION_LIFESPAN_SECONDS=86400
FRONTEND_PUBLIC_URL=https://app.example.com
```

When the API reaches Keycloak through an internal network address, put that
address in `KEYCLOAK_ADMIN_BASE_URL`. Keep the public realm URL in
`OIDC_ISSUER_URL` because it must match the token issuer exactly.

Restart the API and frontend:

```bash
make docker-up-build
```

## 10. Bootstrap The First Tenant Administrator

The API must be running with Keycloak configuration before this command:

```bash
make bootstrap-admin TENANT_ID=acme TENANT_NAME="Acme" ADMIN_EMAIL=admin@acme.com ADMIN_NAME="Acme Admin"
```

The command creates the tenant when needed, creates the Keycloak identity,
persists the PostgreSQL membership and admin role, and sends the setup email.
It rolls back incomplete identity and membership records when a later step
fails.

The administrator completes email verification and password setup, signs in,
opens Users, and invites additional `viewer`, `user`, or `admin` members.

## Local Development

`AUTH_PROVIDER=local` keeps the existing local bearer-token flow for tests and
backend development. The Users screen remains visible to the local admin, but
invitation submission is disabled because local mode is not a credential
provider. Never deploy with `AUTH_PROVIDER=local`.

## References

- [Keycloak Server Administration Guide](https://www.keycloak.org/docs/latest/server_admin/)
- [Keycloak Admin REST API](https://www.keycloak.org/docs-api/latest/rest-api/index.html)
- [Google OAuth 2.0 Web Server Applications](https://developers.google.com/identity/protocols/oauth2/web-server)
- [Creating a GitHub OAuth app](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/creating-an-oauth-app)
