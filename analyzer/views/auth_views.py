"""
Authentication views for user login, logout, and registration.
"""

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import (authenticate, login, logout,
                                 update_session_auth_hash)
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import (AuthenticationForm, PasswordChangeForm,
                                       PasswordResetForm, SetPasswordForm,
                                       UserCreationForm)
from django.contrib.auth.models import User
from django.contrib.auth.tokens import default_token_generator
from django.core.mail import send_mail
from django.shortcuts import redirect, render
from django.template.loader import render_to_string
from django.utils.encoding import force_bytes, force_str
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from django.views.decorators.cache import never_cache
from django.views.decorators.http import require_http_methods
from django_ratelimit.decorators import ratelimit


@ratelimit(key="ip", rate="5/5m", method="POST", block=True)
@never_cache
def login_view(request):
    """
    Handles the login view with redirect support.

    Args:
        request: The HTTP request object.

    Returns:
        HttpResponse: The HTTP response object.
    """
    # Redirect authenticated users
    if request.user.is_authenticated:
        return redirect("index")

    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, f"Welcome back, {user.username}!")

            # GA4 one-shot event (read+popped by context processor on next render)
            request.session["_pending_gtag_event"] = "user_login"
            request.session["_pending_gtag_params"] = {"method": "password"}

            next_url = request.GET.get("next") or request.POST.get("next")
            if next_url:
                return redirect(next_url)
            return redirect("index")
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()

    return render(
        request,
        "analyzer/login.html",
        {"form": form, "next": request.GET.get("next", "")},
    )


@require_http_methods(["GET", "POST"])
def logout_view(request):
    """
    Handles the logout view.

    Args:
        request: The HTTP request object.

    Returns:
        HttpResponse: The HTTP response object.
    """
    if request.user.is_authenticated:
        username = request.user.username
        logout(request)
        messages.info(request, f"You have been logged out successfully, {username}.")
        # GA4 one-shot event (logout flushes the session; the new empty session carries the flag)
        request.session["_pending_gtag_event"] = "user_logout"
    return redirect("login")


@ratelimit(key="ip", rate="3/h", method="POST", block=True)
def register_view(request):
    """
    Handles the registration view.

    Args:
        request: The HTTP request object.

    Returns:
        HttpResponse: The HTTP response object.
    """
    # Redirect authenticated users
    if request.user.is_authenticated:
        return redirect("index")

    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get("username")
            raw_password = form.cleaned_data.get("password1")
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            messages.success(
                request,
                f"Welcome to QueryGrade, {username}! Your account has been created.",
            )
            # GA4 one-shot event (replaces legacy ?signup=1 URL flag for new registrations)
            request.session["_pending_gtag_event"] = "sign_up"
            request.session["_pending_gtag_params"] = {"method": "email"}
            return redirect("index")
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = UserCreationForm()

    return render(request, "analyzer/register.html", {"form": form})


@ratelimit(key="ip", rate="3/h", method="POST", block=True)
@never_cache
def password_reset_request(request):
    """
    Handles password reset request.

    Args:
        request: The HTTP request object.

    Returns:
        HttpResponse: The HTTP response object.
    """
    if request.method == "POST":
        form = PasswordResetForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data["email"]
            users = User.objects.filter(email=email)

            if users.exists():
                for user in users:
                    # Generate password reset token
                    token = default_token_generator.make_token(user)
                    uid = urlsafe_base64_encode(force_bytes(user.pk))

                    # Build reset URL
                    reset_url = request.build_absolute_uri(
                        f"/password-reset-confirm/{uid}/{token}/"
                    )

                    # Send email (in production, use proper email backend)
                    subject = "QueryGrade - Password Reset Request"
                    message = render_to_string(
                        "analyzer/password_reset_email.html",
                        {
                            "user": user,
                            "reset_url": reset_url,
                        },
                    )

                    try:
                        send_mail(
                            subject,
                            message,
                            settings.DEFAULT_FROM_EMAIL,
                            [user.email],
                            fail_silently=False,
                        )
                    except Exception as e:
                        # In development, just show the reset URL
                        messages.warning(
                            request, f"Email not configured. Reset URL: {reset_url}"
                        )

            messages.success(
                request,
                "If an account exists with that email, a password reset link has been sent.",
            )
            request.session["_pending_gtag_event"] = "password_reset_request"
            return redirect("login")
    else:
        form = PasswordResetForm()

    return render(request, "analyzer/password_reset.html", {"form": form})


@never_cache
def password_reset_confirm(request, uidb64, token):
    """
    Handles password reset confirmation.

    Args:
        request: The HTTP request object.
        uidb64: Base64 encoded user ID.
        token: Password reset token.

    Returns:
        HttpResponse: The HTTP response object.
    """
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None

    if user is not None and default_token_generator.check_token(user, token):
        if request.method == "POST":
            form = SetPasswordForm(user, request.POST)
            if form.is_valid():
                form.save()
                messages.success(
                    request,
                    "Your password has been reset successfully. You can now log in.",
                )
                request.session["_pending_gtag_event"] = "password_reset_confirm"
                return redirect("login")
        else:
            form = SetPasswordForm(user)

        return render(
            request,
            "analyzer/password_reset_confirm.html",
            {"form": form, "validlink": True},
        )
    else:
        messages.error(request, "The password reset link is invalid or has expired.")
        return render(
            request, "analyzer/password_reset_confirm.html", {"validlink": False}
        )


@login_required
@never_cache
def password_change(request):
    """
    Handles password change for logged-in users.

    Args:
        request: The HTTP request object.

    Returns:
        HttpResponse: The HTTP response object.
    """
    if request.method == "POST":
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            # Keep user logged in after password change
            update_session_auth_hash(request, user)
            messages.success(request, "Your password has been changed successfully.")
            request.session["_pending_gtag_event"] = "password_change"
            return redirect("account")
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = PasswordChangeForm(request.user)

    return render(request, "analyzer/password_change.html", {"form": form})


@login_required
def account_view(request):
    """
    User account management page.

    Args:
        request: The HTTP request object.

    Returns:
        HttpResponse: The HTTP response object.
    """
    # Get user statistics
    from analyzer.models import QueryFeedback, UserQueryHistory

    query_count = UserQueryHistory.objects.filter(user=request.user).count()

    # QueryFeedback is related through user_history, not directly to user
    feedback_count = QueryFeedback.objects.filter(
        user_history__user=request.user
    ).count()

    # Get recent queries with their analyses
    recent_queries = (
        UserQueryHistory.objects.filter(user=request.user)
        .select_related("query")
        .order_by("-submitted_at")[:5]
    )

    # Get analyses for these queries
    for history in recent_queries:
        try:
            from analyzer.models import QueryAnalysis

            history.analysis = QueryAnalysis.objects.filter(query=history.query).first()
        except Exception:
            history.analysis = None

    context = {
        "query_count": query_count,
        "feedback_count": feedback_count,
        "recent_queries": recent_queries,
    }

    return render(request, "analyzer/account.html", context)
