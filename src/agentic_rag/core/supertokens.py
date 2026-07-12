import logging

from supertokens_python import InputAppInfo, SupertokensConfig, init
from supertokens_python.ingredients.emaildelivery.types import (
    EmailDeliveryConfig,
    SMTPSettings,
    SMTPSettingsFrom,
)
from supertokens_python.recipe import (
    emailpassword,
    emailverification,
    session,
    thirdparty,
)
from supertokens_python.recipe.emailpassword.emaildelivery.services.smtp import (
    SMTPService as EmailPasswordSMTPService,
)
from supertokens_python.recipe.emailverification.emaildelivery.services.smtp import (
    SMTPService as EmailVerificationSMTPService,
)
from supertokens_python.recipe.thirdparty.provider import (
    ProviderClientConfig,
    ProviderConfig,
    ProviderInput,
)
from agentic_rag.shared.config import settings


logger = logging.getLogger(__name__)

_supertokens_initialized = False


def initialize_supertokens() -> bool:
    global _supertokens_initialized

    if settings.auth_provider != "supertokens":
        return False
    if _supertokens_initialized:
        return True
    if not settings.supertokens_api_key.strip():
        raise RuntimeError(
            "SUPERTOKENS_API_KEY is required when AUTH_PROVIDER=supertokens."
        )

    smtp_from = SMTPSettingsFrom(
        name=settings.email_from_name,
        email=settings.email_from_address,
    )
    smtp_settings = SMTPSettings(
        host=settings.smtp_host,
        port=settings.smtp_port,
        from_=smtp_from,
        username=settings.smtp_username or None,
        password=settings.smtp_password or None,
        secure=settings.smtp_use_tls,
    )

    password_email_delivery = None
    verification_email_delivery = None
    if settings.email_delivery_provider == "smtp":
        password_email_delivery = EmailDeliveryConfig(
            service=EmailPasswordSMTPService(smtp_settings)
        )
        verification_email_delivery = EmailDeliveryConfig(
            service=EmailVerificationSMTPService(smtp_settings)
        )

    identity_providers: list[ProviderInput] = []
    if settings.google_client_id and settings.google_client_secret:
        identity_providers.append(
            ProviderInput(
                config=ProviderConfig(
                    third_party_id="google",
                    clients=[
                        ProviderClientConfig(
                            client_id=settings.google_client_id,
                            client_secret=settings.google_client_secret,
                            scope=["openid", "email", "profile"],
                        )
                    ],
                    require_email=True,
                )
            )
        )
    if settings.github_client_id and settings.github_client_secret:
        identity_providers.append(
            ProviderInput(
                config=ProviderConfig(
                    third_party_id="github",
                    clients=[
                        ProviderClientConfig(
                            client_id=settings.github_client_id,
                            client_secret=settings.github_client_secret,
                            scope=["read:user", "user:email"],
                        )
                    ],
                    require_email=True,
                )
            )
        )

    init(
        app_info=InputAppInfo(
            app_name=settings.supertokens_app_name,
            api_domain=settings.supertokens_api_domain,
            website_domain=settings.supertokens_website_domain,
            api_base_path=settings.supertokens_api_base_path,
            website_base_path=settings.supertokens_website_base_path,
        ),
        framework="fastapi",
        supertokens_config=SupertokensConfig(
            connection_uri=settings.supertokens_connection_uri,
            api_key=settings.supertokens_api_key,
        ),
        recipe_list=[
            session.init(
                cookie_domain=settings.supertokens_cookie_domain or None,
                cookie_secure=settings.supertokens_cookie_secure,
                cookie_same_site=settings.supertokens_cookie_same_site,
                anti_csrf="VIA_CUSTOM_HEADER",
            ),
            emailpassword.init(email_delivery=password_email_delivery),
            thirdparty.init(
                sign_in_and_up_feature=thirdparty.SignInAndUpFeature(
                    providers=identity_providers
                )
            ),
            emailverification.init(
                mode="REQUIRED",
                email_delivery=verification_email_delivery,
            ),
        ],
        telemetry=False,
    )

    _supertokens_initialized = True
    logger.info(
        f"[Auth] SuperTokens initialized core={settings.supertokens_connection_uri} "
        f"social_providers={len(identity_providers)}"
    )
    return True
