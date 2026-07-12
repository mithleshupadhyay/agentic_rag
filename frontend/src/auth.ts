import {
  Auth0Client,
  User,
  createAuth0Client,
} from "@auth0/auth0-spa-js";
import SuperTokens from "supertokens-web-js";
import EmailPassword from "supertokens-web-js/recipe/emailpassword";
import EmailVerification from "supertokens-web-js/recipe/emailverification";
import Session from "supertokens-web-js/recipe/session";
import ThirdParty from "supertokens-web-js/recipe/thirdparty";


export type AuthConfiguration = {
  mode: "local" | "supertokens" | "auth0";
  provider: string;
  issuer_url: string | null;
  audience: string | null;
  client_id: string | null;
  scope: string;
  identity_connections: string[];
  api_base_path: string;
  website_base_path: string;
  public_tenant_signup_enabled: boolean;
  social_providers: string[];
};

export type Auth0Session = {
  accessToken: string;
  profile: User;
};

let auth0Client: Auth0Client | null = null;
let auth0ClientConfiguration = "";
let superTokensConfiguration = "";


export async function getAuthenticationErrorMessage(error: unknown): Promise<string> {
  if (error instanceof Response) {
    let responseDetail = "";
    try {
      const payload = (await error.clone().json()) as {
        detail?: unknown;
        message?: unknown;
        reason?: unknown;
      };
      const candidate = payload.detail ?? payload.message ?? payload.reason;
      if (typeof candidate === "string" && candidate.trim()) {
        responseDetail = candidate.trim();
      }
    } catch {
      try {
        responseDetail = (await error.clone().text()).trim();
      } catch {
        responseDetail = "";
      }
    }

    if (error.status >= 500) {
      return "The authentication service is temporarily unavailable. Please retry in a moment.";
    }
    if (error.status === 429) {
      return "Too many authentication attempts. Wait a moment before trying again.";
    }
    if (responseDetail && responseDetail !== "[object Object]") {
      return responseDetail;
    }
    return `Authentication request failed with status ${error.status}.`;
  }

  if (error instanceof Error && error.message.trim()) {
    return error.message;
  }
  if (typeof error === "string" && error.trim()) {
    return error;
  }
  if (error && typeof error === "object") {
    const payload = error as {
      detail?: unknown;
      message?: unknown;
      reason?: unknown;
    };
    const candidate = payload.detail ?? payload.message ?? payload.reason;
    if (typeof candidate === "string" && candidate.trim()) {
      return candidate.trim();
    }
  }
  return "Authentication could not be completed. Please try again.";
}


export async function loadAuthConfiguration(
  apiBaseUrl: string,
): Promise<AuthConfiguration> {
  const baseUrl = apiBaseUrl.trim().replace(/\/$/, "");
  if (!baseUrl) {
    throw new Error("API URL is required before authentication can start.");
  }

  let response: Response;
  try {
    response = await fetch(`${baseUrl}/auth/config`, {
      headers: {
        Accept: "application/json",
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Could not load authentication configuration: ${message}`);
  }

  if (!response.ok) {
    let detail = response.statusText || "Authentication configuration failed";
    try {
      const payload = (await response.json()) as { detail?: unknown };
      if (typeof payload.detail === "string") {
        detail = payload.detail;
      }
    } catch {
      detail = response.statusText || detail;
    }
    throw new Error(
      `Authentication configuration returned ${response.status}: ${detail}`,
    );
  }

  return (await response.json()) as AuthConfiguration;
}


export function configureSuperTokens(
  apiBaseUrl: string,
  configuration: AuthConfiguration,
): void {
  if (configuration.mode !== "supertokens") {
    throw new Error("SuperTokens authentication is not configured.");
  }

  const resolvedApiUrl = new URL(apiBaseUrl, window.location.origin);
  const apiPrefix = resolvedApiUrl.pathname.replace(/\/$/, "");
  const apiBasePath = `${apiPrefix}${configuration.api_base_path}`;
  const configurationKey = JSON.stringify({
    apiDomain: resolvedApiUrl.origin,
    apiBasePath,
    websiteDomain: window.location.origin,
    websiteBasePath: configuration.website_base_path,
  });
  if (configurationKey === superTokensConfiguration) {
    return;
  }
  if (superTokensConfiguration) {
    throw new Error("The authentication API URL cannot change without reloading the page.");
  }

  SuperTokens.init({
    appInfo: {
      appName: "Agentic RAG",
      apiDomain: resolvedApiUrl.origin,
      apiBasePath,
    },
    recipeList: [
      Session.init(),
      EmailPassword.init(),
      ThirdParty.init(),
      EmailVerification.init(),
    ],
  });
  superTokensConfiguration = configurationKey;
}


export async function restoreSuperTokensSession(
  apiBaseUrl: string,
  configuration: AuthConfiguration,
): Promise<boolean> {
  configureSuperTokens(apiBaseUrl, configuration);
  const callbackPath = `${configuration.website_base_path}/callback`;
  if (window.location.pathname.startsWith(callbackPath)) {
    const result = await ThirdParty.signInAndUp();
    if (result.status !== "OK") {
      throw new Error(
        result.status === "NO_EMAIL_GIVEN_BY_PROVIDER"
          ? "The identity provider did not return a recovery email address."
          : result.reason,
      );
    }
    window.history.replaceState({}, document.title, "/");
  }

  const verificationPath = `${configuration.website_base_path}/verify-email`;
  if (
    window.location.pathname === verificationPath &&
    EmailVerification.getEmailVerificationTokenFromURL()
  ) {
    const result = await EmailVerification.verifyEmail();
    if (result.status !== "OK") {
      throw new Error("The email verification link is invalid or has expired.");
    }
    window.history.replaceState({}, document.title, "/");
  }
  return Session.doesSessionExist();
}


export async function signInWithEmailPassword(
  email: string,
  password: string,
): Promise<void> {
  const result = await EmailPassword.signIn({
    formFields: [
      { id: "email", value: email.trim() },
      { id: "password", value: password },
    ],
  });
  if (result.status === "WRONG_CREDENTIALS_ERROR") {
    throw new Error("Email or password is incorrect.");
  }
  if (result.status === "FIELD_ERROR") {
    throw new Error(result.formFields.map((field) => field.error).join(" "));
  }
  if (result.status === "SIGN_IN_NOT_ALLOWED") {
    throw new Error(result.reason);
  }
}


export async function signUpWithEmailPassword(
  email: string,
  password: string,
): Promise<void> {
  const result = await EmailPassword.signUp({
    formFields: [
      { id: "email", value: email.trim() },
      { id: "password", value: password },
    ],
  });
  if (result.status === "FIELD_ERROR") {
    throw new Error(result.formFields.map((field) => field.error).join(" "));
  }
  if (result.status === "SIGN_UP_NOT_ALLOWED") {
    throw new Error(result.reason);
  }
  await EmailVerification.sendVerificationEmail();
}


export async function startSuperTokensSocialLogin(provider: string): Promise<void> {
  const redirectUrl = await ThirdParty.getAuthorisationURLWithQueryParamsAndSetState({
    thirdPartyId: provider,
    frontendRedirectURI: `${window.location.origin}/auth/callback`,
  });
  window.location.assign(redirectUrl);
}


export async function sendPasswordReset(email: string): Promise<void> {
  const result = await EmailPassword.sendPasswordResetEmail({
    formFields: [{ id: "email", value: email.trim() }],
  });
  if (result.status === "FIELD_ERROR") {
    throw new Error(result.formFields.map((field) => field.error).join(" "));
  }
  if (result.status === "PASSWORD_RESET_NOT_ALLOWED") {
    throw new Error(result.reason);
  }
}


export async function submitPasswordReset(password: string): Promise<void> {
  const result = await EmailPassword.submitNewPassword({
    formFields: [{ id: "password", value: password }],
  });
  if (result.status === "RESET_PASSWORD_INVALID_TOKEN_ERROR") {
    throw new Error("The password reset link is invalid or has expired.");
  }
  if (result.status === "FIELD_ERROR") {
    throw new Error(result.formFields.map((field) => field.error).join(" "));
  }
}


export async function signOutSuperTokens(): Promise<void> {
  await Session.signOut();
}


export async function configureAuth0Client(
  configuration: AuthConfiguration,
): Promise<Auth0Client> {
  if (
    configuration.mode !== "auth0" ||
    !configuration.issuer_url ||
    !configuration.audience ||
    !configuration.client_id
  ) {
    throw new Error(
      "Auth0 domain, API audience, and browser client ID are required.",
    );
  }

  const issuerUrl = new URL(configuration.issuer_url);
  const configurationKey = JSON.stringify({
    domain: issuerUrl.host,
    audience: configuration.audience,
    clientId: configuration.client_id,
    scope: configuration.scope,
  });
  if (auth0Client && auth0ClientConfiguration === configurationKey) {
    return auth0Client;
  }

  auth0Client = await createAuth0Client({
    domain: issuerUrl.host,
    clientId: configuration.client_id,
    authorizationParams: {
      audience: configuration.audience,
      redirect_uri: `${window.location.origin}/auth/callback`,
      scope: configuration.scope,
    },
    cacheLocation: "memory",
    useRefreshTokens: false,
  });
  auth0ClientConfiguration = configurationKey;
  return auth0Client;
}


export async function restoreAuth0Session(
  configuration: AuthConfiguration,
): Promise<Auth0Session | null> {
  const client = await configureAuth0Client(configuration);
  const searchParameters = new URLSearchParams(window.location.search);

  if (
    window.location.pathname === "/auth/callback" &&
    searchParameters.has("code") &&
    searchParameters.has("state")
  ) {
    await client.handleRedirectCallback();
    window.history.replaceState({}, document.title, "/");
  } else if (
    searchParameters.has("invitation") &&
    searchParameters.has("organization")
  ) {
    const invitation = searchParameters.get("invitation");
    const organization = searchParameters.get("organization");
    if (!invitation || !organization) {
      throw new Error("The Auth0 organization invitation link is incomplete.");
    }

    await client.loginWithRedirect({
      authorizationParams: {
        invitation,
        organization,
      },
      appState: { returnTo: "/" },
    });
    return null;
  }

  const isAuthenticated = await client.isAuthenticated();
  if (!isAuthenticated) {
    return null;
  }

  const accessToken = await client.getTokenSilently();
  const profile = await client.getUser();
  if (!profile) {
    throw new Error("Auth0 did not return the signed-in user profile.");
  }

  return { accessToken, profile };
}


export async function startAuth0Login(
  configuration: AuthConfiguration,
  identityConnection?: string,
): Promise<void> {
  const client = await configureAuth0Client(configuration);
  const connection = identityConnection?.trim();

  await client.loginWithRedirect({
    authorizationParams: connection ? { connection } : {},
    appState: { returnTo: "/" },
  });
}


export async function startAuth0Logout(
  configuration: AuthConfiguration,
): Promise<void> {
  const client = await configureAuth0Client(configuration);
  await client.logout({
    logoutParams: {
      returnTo: window.location.origin,
    },
  });
}


export async function subscribeToAuth0Session(
  configuration: AuthConfiguration,
  onSessionChanged: (session: Auth0Session | null) => void,
): Promise<() => void> {
  const client = await configureAuth0Client(configuration);
  let refreshInProgress = false;

  const refreshSession = async () => {
    if (refreshInProgress) {
      return;
    }
    refreshInProgress = true;

    try {
      if (!(await client.isAuthenticated())) {
        onSessionChanged(null);
        return;
      }

      const accessToken = await client.getTokenSilently();
      const profile = await client.getUser();
      if (!profile) {
        onSessionChanged(null);
        return;
      }
      onSessionChanged({ accessToken, profile });
    } catch {
      onSessionChanged(null);
    } finally {
      refreshInProgress = false;
    }
  };

  const intervalId = window.setInterval(() => {
    void refreshSession();
  }, 60000);

  return () => {
    window.clearInterval(intervalId);
  };
}
