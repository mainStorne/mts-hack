import { Trans } from "@lingui/react/macro";
import { createRootRoute, Link, Outlet } from "@tanstack/react-router";

export const Route = createRootRoute({
  component: Outlet,

  // 404 page
  notFoundComponent: () => (
    <div className="flex h-full flex-col items-center justify-center">
      <h1 className="mb-4 text-4xl font-bold">
        <Trans>404 / not found</Trans>
      </h1>
      <Link to="/" className="selected">
        <Trans>Go to the chat</Trans>
      </Link>
    </div>
  ),
});
